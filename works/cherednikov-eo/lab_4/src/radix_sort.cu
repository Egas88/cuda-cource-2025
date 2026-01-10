#include <stdio.h>
#include <vector>
#include <chrono>
#include <stdint.h>
#include <cuda_runtime.h>
#include "error_check.cuh"

// Макросы для настройки работы с shared memory и банками памяти
#define THREADS_PER_BLOCK 256
#define ELEMENTS_PER_BLOCK (2*THREADS_PER_BLOCK)
#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) (((n) >> LOG_NUM_BANKS))
static constexpr int RADIX_BITS = 8;
static constexpr int RADIX = 1 << RADIX_BITS;


 // выполнение эксклюзивного префиксного суммирования в рамках одного блока
__global__ void scanSingleBlockExclusive(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, uint32_t* __restrict__ blockSums, int nTotal) {
    extern __shared__ uint32_t temp[];

    int t = threadIdx.x;
    int blockOffset = blockIdx.x * ELEMENTS_PER_BLOCK;

    int ai = t;
    int bi = t + (ELEMENTS_PER_BLOCK / 2);

    // Вычисление смещений для избежания конфликтов банков памяти
    int bankA = CONFLICT_FREE_OFFSET(ai);
    int bankB = CONFLICT_FREE_OFFSET(bi);

    temp[ai + bankA] = (blockOffset + ai < nTotal) ? in[blockOffset + ai] : 0u;
    temp[bi + bankB] = (blockOffset + bi < nTotal) ? in[blockOffset + bi] : 0u;

    int offset = 1;
    for (int d = ELEMENTS_PER_BLOCK >> 1; d > 0; d >>= 1) {
        __syncthreads();
        if (t < d) {
            int ai2 = offset * (2 * t + 1) - 1;
            int bi2 = offset * (2 * t + 2) - 1;
            ai2 += CONFLICT_FREE_OFFSET(ai2);
            bi2 += CONFLICT_FREE_OFFSET(bi2);
            temp[bi2] += temp[ai2];
        }
        offset <<= 1;
    }
    __syncthreads();


    if (t == 0) {
        int last = ELEMENTS_PER_BLOCK - 1;
        int lastIdx = last + CONFLICT_FREE_OFFSET(last);
        blockSums[blockIdx.x] = temp[lastIdx];
        temp[lastIdx] = 0u;
    }

    // Фаза нисходящего обхода, вычисление префиксных сумм
    for (int d = 1; d < ELEMENTS_PER_BLOCK; d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (t < d) {
            int ai2 = offset * (2 * t + 1) - 1;
            int bi2 = offset * (2 * t + 2) - 1;
            ai2 += CONFLICT_FREE_OFFSET(ai2);
            bi2 += CONFLICT_FREE_OFFSET(bi2);

            uint32_t x = temp[ai2];
            temp[ai2] = temp[bi2];
            temp[bi2] += x;
        }
    }
    __syncthreads();

    // Записываем всё в глобальгую память
    if (blockOffset + ai < nTotal) out[blockOffset + ai] = temp[ai + bankA];
    if (blockOffset + bi < nTotal) out[blockOffset + bi] = temp[bi + bankB];
}


// добавление префиксной суммы блока к результатам сканирования
__global__ void addBlockPrefixToScanned(uint32_t* __restrict__ scanned, const uint32_t* __restrict__ blockPrefix, int nTotal) {
    int blockOffset = blockIdx.x * ELEMENTS_PER_BLOCK;
    uint32_t add = blockPrefix[blockIdx.x];

    int t = threadIdx.x;
    int ai = blockOffset + t;
    int bi = blockOffset + t + (ELEMENTS_PER_BLOCK / 2);

    // Добавление префиксной суммы к каждому элементу блока
    if (ai < nTotal) scanned[ai] += add;
    if (bi < nTotal) scanned[bi] += add;
}


//Функция для выполнения эксклюзивного префиксного суммирования (uint32_t)
static void scanExclusive_u32(uint32_t* d_in, uint32_t* d_out, int n) {
    if (n <= 0) return;

    int numBlocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

    uint32_t* d_blockSums = nullptr;   // Массив для хранения сумм каждого блока
    uint32_t* d_blockPrefix = nullptr;  // Массив для хранения префиксных сумм блоков

    CUDA_CHECK(cudaMalloc(&d_blockSums,   numBlocks * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_blockPrefix, numBlocks * sizeof(uint32_t)));

    int sharedSize = (ELEMENTS_PER_BLOCK + CONFLICT_FREE_OFFSET(ELEMENTS_PER_BLOCK)) * (int)sizeof(uint32_t);

    scanSingleBlockExclusive<<<numBlocks, THREADS_PER_BLOCK, sharedSize>>>(d_in, d_out, d_blockSums, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    if (numBlocks > 1) {
        scanExclusive_u32(d_blockSums, d_blockPrefix, numBlocks);
    } else {
        CUDA_CHECK(cudaMemset(d_blockPrefix, 0, sizeof(uint32_t)));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    addBlockPrefixToScanned<<<numBlocks, THREADS_PER_BLOCK>>>(d_out, d_blockPrefix, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_blockPrefix));
}


// извлечение определенного бита из каждого элемента массива
template<typename T>
__global__ void extractDigitKernel(const T* __restrict__ in, uint32_t* __restrict__ digitOut, int n, int shift) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
		using U = typename std::make_unsigned<T>::type;
        U v = (U)in[idx];
        digitOut[idx] = (uint32_t)((v >> shift) & (RADIX - 1));
    }
}


__global__ void flagEqualsKernel(const uint32_t* __restrict__ digits, uint32_t* __restrict__ flags, int n, uint32_t k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) flags[idx] = (digits[idx] == k) ? 1u : 0u;
}


//распределение элементов по новым позициям на основе битовой маски
template<typename T>
__global__ void scatterByDigitKernel(const T* __restrict__ in, T* __restrict__ out, const uint32_t* __restrict__ digits, const uint32_t* __restrict__ scanFlags, int n, uint32_t k, uint32_t baseOffsetK) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (digits[idx] == k) {
            uint32_t pos = baseOffsetK + scanFlags[idx];
            out[pos] = in[idx];
        }
    }
}


//инвертирование знакового бита в элементах массива
template<typename U>
__global__ void flipSignBitKernel(U* data, int n, U mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] ^= mask; // Инвертирование знакового бита через XOR
}


template<typename T>
static void radixSortByScanDigits(T* d_in, T* d_out, int n) {
    if (n <= 0) return;
    T* src = d_in;
    T* dst = d_out;

    uint32_t* d_digits = nullptr;
    uint32_t* d_flags  = nullptr;
    uint32_t* d_scan   = nullptr;
    CUDA_CHECK(cudaMalloc(&d_digits, (size_t)n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_flags,  (size_t)n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_scan,   (size_t)n * sizeof(uint32_t)));

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBits = (int)(sizeof(T) * 8);
    int passes  = (numBits + RADIX_BITS - 1) / RADIX_BITS;

    for (int pass = 0; pass < passes; ++pass) {
        int shift = pass * RADIX_BITS;

        extractDigitKernel<T><<<blocks, THREADS_PER_BLOCK>>>(src, d_digits, n, shift);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t counts[RADIX];
        uint32_t baseOffset[RADIX];

        for (uint32_t k = 0; k < RADIX; ++k) {
            flagEqualsKernel<<<blocks, THREADS_PER_BLOCK>>>(d_digits, d_flags, n, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            scanExclusive_u32(d_flags, d_scan, n);

            uint32_t lastScan = 0, lastFlag = 0;
            CUDA_CHECK(cudaMemcpy(&lastScan, d_scan + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&lastFlag, d_flags + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
            counts[k] = lastScan + lastFlag;
        }

        uint32_t sum = 0;
        for (int k = 0; k < RADIX; ++k) {
            baseOffset[k] = sum;
            sum += counts[k];
        }

        for (uint32_t k = 0; k < RADIX; ++k) {
            flagEqualsKernel<<<blocks, THREADS_PER_BLOCK>>>(d_digits, d_flags, n, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            scanExclusive_u32(d_flags, d_scan, n);

            scatterByDigitKernel<T><<<blocks, THREADS_PER_BLOCK>>>(
                src, dst, d_digits, d_scan, n, k, baseOffset[k]
            );
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        T* tmp = src; src = dst; dst = tmp;
    }

    if (src != d_out) {
        CUDA_CHECK(cudaMemcpy(d_out, src, (size_t)n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_digits));
    CUDA_CHECK(cudaFree(d_flags));
    CUDA_CHECK(cudaFree(d_scan));
}


void radix_sort_uint32(uint32_t* d_in, uint32_t* d_out, int n) {
    radixSortByScanDigits<uint32_t>(d_in, d_out, n);
}


void radix_sort_uint64(uint64_t* d_in, uint64_t* d_out, int n) {
    radixSortByScanDigits<uint64_t>(d_in, d_out, n);
}


void radix_sort_int32(int32_t* d_in, int32_t* d_out, int n) {
    uint32_t* u_in  = reinterpret_cast<uint32_t*>(d_in);
    uint32_t* u_out = reinterpret_cast<uint32_t*>(d_out);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    flipSignBitKernel<uint32_t><<<blocks, THREADS_PER_BLOCK>>>(u_in, n, 0x80000000u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    radixSortByScanDigits<uint32_t>(u_in, u_out, n);

    flipSignBitKernel<uint32_t><<<blocks, THREADS_PER_BLOCK>>>(u_out, n, 0x80000000u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}


void radix_sort_int64(int64_t* d_in, int64_t* d_out, int n) {
    uint64_t* u_in  = reinterpret_cast<uint64_t*>(d_in);
    uint64_t* u_out = reinterpret_cast<uint64_t*>(d_out);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    flipSignBitKernel<uint64_t><<<blocks, THREADS_PER_BLOCK>>>(u_in, n, 0x8000000000000000ull);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    radixSortByScanDigits<uint64_t>(u_in, u_out, n);

    flipSignBitKernel<uint64_t><<<blocks, THREADS_PER_BLOCK>>>(u_out, n, 0x8000000000000000ull);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}