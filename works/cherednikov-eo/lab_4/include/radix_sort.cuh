#ifndef CUDA_COURCE_2025_RADIX_SORT_CUH
#define CUDA_COURCE_2025_RADIX_SORT_CUH

#include <cuda_runtime.h>
#include <cstdint>

constexpr int THREADS_PER_BLOCK = 256; // потоки на блок;
constexpr int BITS_PER_PASS = 8; // сколько бит обрабатывается;
constexpr int RADIX = 1 << BITS_PER_PASS;  // количество разрядов;

// Radix Sort для 32-bit unsigned integers
void radix_sort_uint32(uint32_t* d_data, size_t n);

// Radix Sort для 64-bit unsigned integers
void radix_sort_uint64(uint64_t* d_data, size_t n);

// Radix Sort для 32-bit signed integers
void radix_sort_int32(int32_t* d_data, size_t n);

// Radix Sort для 64-bit signed integers
void radix_sort_int64(int64_t* d_data, size_t n);


template<typename T>
struct RadixMethods;

template<>
struct RadixMethods<int32_t> {
    static constexpr bool is_signed = true;
    static constexpr const char* name = "int32_t";
    static void sort(int32_t* d, size_t n) { radix_sort_int32(d, n); }
};

template<>
struct RadixMethods<uint32_t> {
    static constexpr bool is_signed = false;
    static constexpr const char* name = "uint32_t";
    static void sort(uint32_t* d, size_t n) { radix_sort_uint32(d, n); }
};

template<>
struct RadixMethods<int64_t> {
    static constexpr bool is_signed = true;
    static constexpr const char* name = "int64_t";
    static void sort(int64_t* d, size_t n) { radix_sort_int64(d, n); }
};

template<>
struct RadixMethods<uint64_t> {
    static constexpr bool is_signed = false;
    static constexpr const char* name = "uint64_t";
    static void sort(uint64_t* d, size_t n) { radix_sort_uint64(d, n); }
};


#endif //CUDA_COURCE_2025_RADIX_SORT_CUH