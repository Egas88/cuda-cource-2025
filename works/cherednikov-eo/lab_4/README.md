# Lab 4: Radix Sort - Поразрядная сортировка

## Описание

Реализация алгоритма поразрядной сортировки (Radix Sort) на CUDA для сортировки целых чисел различных типов (32-bit и 64-bit, signed и unsigned).

## Компиляция

```bash
make
```

## Запуск

```bash
./radix
```

или

```bash
make run
```

## Результаты

```
Benchmark: uint32_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.07491s, GPU Thrust=0.00009s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.31x, Radix vs Thrust=0.00x

Benchmark: int32_t[1000] OK
Time: CPU=0.00002s, GPU Radix=0.07653s, GPU Thrust=0.00008s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.28x, Radix vs Thrust=0.00x

Benchmark: uint64_t[1000] OK
Time: CPU=0.00002s, GPU Radix=0.15547s, GPU Thrust=0.00009s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.28x, Radix vs Thrust=0.00x

Benchmark: int64_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.15274s, GPU Thrust=0.00009s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.28x, Radix vs Thrust=0.00x

Benchmark: uint32_t[100000] OK
Time: CPU=0.00400s, GPU Radix=0.07773s, GPU Thrust=0.00026s
Speedup: Radix vs CPU=0.05x, Thrust vs CPU=15.12x, Radix vs Thrust=0.00x

Benchmark: int32_t[100000] OK
Time: CPU=0.00391s, GPU Radix=0.07762s, GPU Thrust=0.00027s
Speedup: Radix vs CPU=0.05x, Thrust vs CPU=14.30x, Radix vs Thrust=0.00x

Benchmark: uint64_t[100000] OK
Time: CPU=0.00394s, GPU Radix=0.15281s, GPU Thrust=0.00035s
Speedup: Radix vs CPU=0.03x, Thrust vs CPU=11.33x, Radix vs Thrust=0.00x

Benchmark: int64_t[100000] OK
Time: CPU=0.00401s, GPU Radix=0.15644s, GPU Thrust=0.00035s
Speedup: Radix vs CPU=0.03x, Thrust vs CPU=11.34x, Radix vs Thrust=0.00x

Benchmark: uint32_t[5000000] OK
Time: CPU=0.18687s, GPU Radix=0.22642s, GPU Thrust=0.00084s
Speedup: Radix vs CPU=0.83x, Thrust vs CPU=223.39x, Radix vs Thrust=0.00x

Benchmark: int32_t[5000000] OK
Time: CPU=0.18628s, GPU Radix=0.23004s, GPU Thrust=0.00085s
Speedup: Radix vs CPU=0.81x, Thrust vs CPU=217.98x, Radix vs Thrust=0.00x

Benchmark: uint64_t[5000000] OK
Time: CPU=0.18856s, GPU Radix=0.46620s, GPU Thrust=0.00221s
Speedup: Radix vs CPU=0.40x, Thrust vs CPU=85.19x, Radix vs Thrust=0.00x

Benchmark: int64_t[5000000] OK
Time: CPU=0.19294s, GPU Radix=0.45226s, GPU Thrust=0.00220s
Speedup: Radix vs CPU=0.43x, Thrust vs CPU=87.61x, Radix vs Thrust=0.00x

Benchmark: uint32_t[10000000] OK
Time: CPU=0.37020s, GPU Radix=0.26837s, GPU Thrust=0.00147s
Speedup: Radix vs CPU=1.38x, Thrust vs CPU=251.24x, Radix vs Thrust=0.01x

Benchmark: int32_t[10000000] OK
Time: CPU=0.37461s, GPU Radix=0.27143s, GPU Thrust=0.00155s
Speedup: Radix vs CPU=1.38x, Thrust vs CPU=241.45x, Radix vs Thrust=0.01x

Benchmark: uint64_t[10000000] OK
Time: CPU=0.36904s, GPU Radix=0.51143s, GPU Thrust=0.00384s
Speedup: Radix vs CPU=0.72x, Thrust vs CPU=96.20x, Radix vs Thrust=0.01x

Benchmark: int64_t[10000000] OK
Time: CPU=0.38026s, GPU Radix=0.50084s, GPU Thrust=0.00389s
Speedup: Radix vs CPU=0.76x, Thrust vs CPU=97.64x, Radix vs Thrust=0.01x
```