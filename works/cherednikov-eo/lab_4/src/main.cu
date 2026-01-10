#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include "error_check.cuh"
#include "radix_sort.cuh"


template<typename T> bool check_sorting(T* data, size_t n) {
    for (size_t i = 1; i < n; i++) {
        if (data[i] < data[i-1]) {
            return false;
        }
    }
    return true;
}


inline void print_results(const char* type_name, size_t n, double cpu_time, float gpu_radix_time, float gpu_thrust_time) {
    printf("Benchmark: %s[%zu] OK\n", type_name, n);

    printf("Time: CPU=%.5fs, GPU Radix=%.5fs, GPU Thrust=%.5fs\n",
           cpu_time, gpu_radix_time, gpu_thrust_time);

    printf("Speedup: Radix vs CPU=%.2fx, Thrust vs CPU=%.2fx, Radix vs Thrust=%.2fx\n\n",
           cpu_time / gpu_radix_time,
           cpu_time / gpu_thrust_time,
           gpu_thrust_time / gpu_radix_time);
}


template<typename T>
void generate_random_data(T* data, size_t n, bool signed_type) {
    for (size_t i = 0; i < n; i++) {
        if (signed_type) {
            // Для signed: положительные и отрицательные числа
            data[i] = (T)((rand() % 2000000) - 1000000);
        } else {
            // Для unsigned: только положительные
            data[i] = (T)(rand() % 1000000);
        }
    }
}


template <typename T>
void free_all(T* h1, T* h2, T* h3, T* h4, T* d) {
    free(h1);
    free(h2);
    free(h3);
    free(h4);
    CUDA_CHECK(cudaFree(d));
}


template <typename T>
double cpuSort(T* data, size_t  n) {
    auto start = std::chrono::high_resolution_clock::now();
    std::sort(data, data + n);
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(end - start).count();
}


template <typename T>
float gpu_radix_sort(T* d_data, T* h_data,size_t n, void (*radix_func)(T*, size_t)) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(T), cudaMemcpyHostToDevice));

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    radix_func(d_data, n);
    cudaEventRecord(stop);

    CUDA_CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(T), cudaMemcpyDeviceToHost));
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}


template <typename T>
float gpu_thrust_sort(T* h_data, size_t n) {
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    thrust::device_vector<T> d_data(n);
    CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(d_data.data()), h_data, n * sizeof(T), cudaMemcpyHostToDevice));

    cudaEventRecord(start);
    thrust::sort(d_data.begin(), d_data.end());
    cudaEventRecord(stop);

    CUDA_CHECK(cudaDeviceSynchronize());
    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}


template<typename T>
void benchmark(size_t n) {
    using method = RadixMethods<T>;
    T* h_data        = (T*)malloc(n * sizeof(T));
    T* h_cpu         = (T*)malloc(n * sizeof(T));
    T* h_gpu         = (T*)malloc(n * sizeof(T));
    T* h_thrust      = (T*)malloc(n * sizeof(T));

    T* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(T)));

    generate_random_data(h_data, n, method::is_signed);
    memcpy(h_cpu, h_data, n * sizeof(T));
    memcpy(h_gpu, h_data, n * sizeof(T));
    memcpy(h_thrust, h_data, n * sizeof(T));

    double cpu_time = cpuSort(h_cpu, n);
    float gpu_radix_time = gpu_radix_sort(d_data, h_gpu, n, method::sort);

    bool correct = check_sorting(h_gpu, n);
    if (!correct) {
        printf("Benchmark: %s[%zu] FAILED\n\n", method::name, n);
        free_all(h_data, h_cpu, h_gpu, h_thrust, d_data);
        return;
    }
    float gpu_thrust_time = gpu_thrust_sort(h_thrust, n);
    print_results(method::name, n, cpu_time, gpu_radix_time, gpu_thrust_time);
    free_all(h_data, h_cpu, h_gpu, h_thrust, d_data);
}


int main() {
    srand(158);

    size_t sizes[] = {1000, 100000, 5000000, 10000000};

    for (size_t n : sizes) {
        benchmark<int32_t>(n);
        benchmark<int64_t>(n);
        benchmark<uint32_t>(n);
        benchmark<uint64_t>(n);
    }
    using method = RadixMethods<int32_t>;

    return 0;
}
