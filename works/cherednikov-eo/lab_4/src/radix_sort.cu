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
__global__ void extractBitKernel(const T* __restrict__ in, uint32_t* __restrict__ bits, int n, int bitPos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Извлечение бита на позиции bitPos из элемента in[idx]
        bits[idx] = (uint32_t)((((uint64_t)in[idx]) >> bitPos) & 1ull);
    }
}


//распределение элементов по новым позициям на основе битовой маски
template<typename T>
__global__ void scatterByBitKernel(const T* __restrict__ in, T* __restrict__ out, const uint32_t* __restrict__ scanOnes, const uint32_t* __restrict__ bits, int n, int numZeros) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t b = bits[idx]; // Текущий бит элемента
        uint32_t onesBefore = scanOnes[idx]; // Количество единиц до текущей позиции

        uint32_t pos = (b == 0)
            ? (uint32_t)idx - onesBefore
            : (uint32_t)numZeros + onesBefore;
        out[pos] = in[idx];
    }
}


//инвертирование знакового бита в элементах массива
template<typename U>
__global__ void flipSignBitKernel(U* data, int n, U mask) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] ^= mask; // Инвертирование знакового бита через XOR
}


template<typename T>
static void radixSortByScanBits(T* d_in, T* d_out, int n) {
    if (n <= 0) return;
    T* src = d_in;
    T* dst = d_out;

    uint32_t* d_bits = nullptr;
    uint32_t* d_scan = nullptr;
    CUDA_CHECK(cudaMalloc(&d_bits, n * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_scan, n * sizeof(uint32_t)));

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int numBits = (int)(sizeof(T) * 8);


    for (int bit = 0; bit < numBits; ++bit) {
        // Извлечение текущего бита из всех элементов
        extractBitKernel<T><<<blocks, THREADS_PER_BLOCK>>>(src, d_bits, n, bit);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Вычисление префиксных сумм для битов (количество единиц до каждой позиции)
        scanExclusive_u32(d_bits, d_scan, n);

        // Вычисление общего количества единиц и нулей
        uint32_t lastScan = 0, lastBit = 0;
        CUDA_CHECK(cudaMemcpy(&lastScan, d_scan + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&lastBit,  d_bits + (n - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost));
        int totalOnes = (int)(lastScan + lastBit);
        int numZeros  = n - totalOnes;

        // Перераспределение элементов на основе битовой маски
        scatterByBitKernel<T><<<blocks, THREADS_PER_BLOCK>>>(src, dst, d_scan, d_bits, n, numZeros);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        T* tmp = src;
		src = dst;
		dst = tmp;
    }

    if (src != d_out) {
        CUDA_CHECK(cudaMemcpy(d_out, src, (size_t)n * sizeof(T), cudaMemcpyDeviceToDevice));
    }

    CUDA_CHECK(cudaFree(d_bits));
    CUDA_CHECK(cudaFree(d_scan));
}


void radix_sort_uint32(uint32_t* d_in, uint32_t* d_out, int n) {
    radixSortByScanBits<uint32_t>(d_in, d_out, n);
}


void radix_sort_uint64(uint64_t* d_in, uint64_t* d_out, int n) {
    radixSortByScanBits<uint64_t>(d_in, d_out, n);
}


void radix_sort_int32(int32_t* d_in, int32_t* d_out, int n) {
    uint32_t* u_in  = reinterpret_cast<uint32_t*>(d_in);
    uint32_t* u_out = reinterpret_cast<uint32_t*>(d_out);

    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    flipSignBitKernel<uint32_t><<<blocks, THREADS_PER_BLOCK>>>(u_in, n, 0x80000000u);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    radixSortByScanBits<uint32_t>(u_in, u_out, n);

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

    radixSortByScanBits<uint64_t>(u_in, u_out, n);

    flipSignBitKernel<uint64_t><<<blocks, THREADS_PER_BLOCK>>>(u_out, n, 0x8000000000000000ull);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}