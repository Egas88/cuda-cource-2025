#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>
#include "error_check.cuh"
#include "radix_sort.cuh"
#include <utility>


template<typename T>
__global__ void histogram_kernel(T* data, size_t n, uint32_t bit_offset, uint32_t* hist) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T value = data[idx];
    uint32_t bin = (value >> bit_offset) & (RADIX - 1);
    atomicAdd(&hist[bin], 1);
}


template<typename T>
__global__ void scatter_kernel(T* input, T* output, size_t n, uint32_t bit_offset, const uint32_t* prefix_sum, uint32_t* offsets) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    T value = input[idx];
    uint32_t bin = (value >> bit_offset) & (RADIX - 1);

    uint32_t pos = atomicAdd(&offsets[bin], 1);
    output[prefix_sum[bin] + pos] = value;
}


// Kernel для prefix sum (exclusive scan)
__global__ void prefix_sum_kernel(uint32_t* hist, uint32_t* prefix_sum, int num_bins) {
    int idx = threadIdx.x;
    if (idx >= num_bins) return;
    uint32_t sum = 0;
    for (int i = 0; i < idx; i++) {
        sum += hist[i];
    }
    prefix_sum[idx] = sum;
}


template<typename T, int TOTAL_BITS>
void radix_sort_unsigned(T* d_data, size_t n) {
    if (n == 0) return;

    T* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, n * sizeof(T)));

    uint32_t* d_hist;
    uint32_t* d_prefix;
    uint32_t* d_offsets;
    CUDA_CHECK(cudaMalloc(&d_offsets, RADIX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_hist, RADIX * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_prefix, RADIX * sizeof(uint32_t)));

    T* input = d_data;
    T* output = d_temp;

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    for (int bit = 0; bit < TOTAL_BITS; bit += BITS_PER_PASS) {
        CUDA_CHECK(cudaMemset(d_hist, 0, RADIX * sizeof(uint32_t)));

        histogram_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, n, bit, d_hist);
        CUDA_CHECK(cudaDeviceSynchronize());

        prefix_sum_kernel<<<1, RADIX>>>(d_hist, d_prefix, RADIX);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(d_offsets,d_prefix,RADIX * sizeof(uint32_t),cudaMemcpyDeviceToDevice));
        scatter_kernel<<<blocks, THREADS_PER_BLOCK>>>(input, output, n, bit, d_prefix, d_offsets);
        CUDA_CHECK(cudaDeviceSynchronize());

        T* tmp = input;
        input = output;
        output = tmp;
    }

    if (input != d_data) {
        CUDA_CHECK(cudaMemcpy(d_data, input, n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_temp));
    CUDA_CHECK(cudaFree(d_hist));
    CUDA_CHECK(cudaFree(d_prefix));
}


// Kernels для конвертации signed <-> unsigned
template<typename S, typename U, U MASK>
__global__ void convert_signed_to_unsigned(S* in, U* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<U>(in[idx]) ^ MASK;
}


template<typename S, typename U, U MASK>
__global__ void convert_unsigned_to_signed(U* in, S* out, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = static_cast<S>(in[idx] ^ MASK);
}


void radix_sort_uint32(uint32_t* d, size_t n) {
    radix_sort_unsigned<uint32_t, 32>(d, n);
}


void radix_sort_uint64(uint64_t* d, size_t n) {
    radix_sort_unsigned<uint64_t, 64>(d, n);
}


void radix_sort_int32(int32_t* d, size_t n) {
    uint32_t* tmp;
    CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(uint32_t)));

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    convert_signed_to_unsigned<int32_t, uint32_t, 0x80000000u>
        <<<blocks, THREADS_PER_BLOCK>>>(d, tmp, n);

    radix_sort_uint32(tmp, n);

    convert_unsigned_to_signed<int32_t, uint32_t, 0x80000000u>
        <<<blocks, THREADS_PER_BLOCK>>>(tmp, d, n);

    CUDA_CHECK(cudaFree(tmp));
}


void radix_sort_int64(int64_t* d, size_t n) {
    uint64_t* tmp;
    CUDA_CHECK(cudaMalloc(&tmp, n * sizeof(uint64_t)));

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    convert_signed_to_unsigned<int64_t, uint64_t, 0x8000000000000000ULL>
        <<<blocks, THREADS_PER_BLOCK>>>(d, tmp, n);

    radix_sort_uint64(tmp, n);

    convert_unsigned_to_signed<int64_t, uint64_t, 0x8000000000000000ULL>
        <<<blocks, THREADS_PER_BLOCK>>>(tmp, d, n);

    CUDA_CHECK(cudaFree(tmp));
}