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
Time: CPU=0.00003s, GPU Radix=0.01203s, GPU Thrust=0.00009s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.33x, Radix vs Thrust=0.01x

Benchmark: int32_t[1000] OK
Time: CPU=0.00002s, GPU Radix=0.01196s, GPU Thrust=0.00008s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.28x, Radix vs Thrust=0.01x

Benchmark: uint64_t[1000] OK
Time: CPU=0.00002s, GPU Radix=0.02360s, GPU Thrust=0.00009s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.29x, Radix vs Thrust=0.00x

Benchmark: int64_t[1000] OK
Time: CPU=0.00003s, GPU Radix=0.02482s, GPU Thrust=0.00009s
Speedup: Radix vs CPU=0.00x, Thrust vs CPU=0.29x, Radix vs Thrust=0.00x

Benchmark: uint32_t[100000] OK
Time: CPU=0.00390s, GPU Radix=0.01241s, GPU Thrust=0.00040s
Speedup: Radix vs CPU=0.31x, Thrust vs CPU=9.85x, Radix vs Thrust=0.03x

Benchmark: int32_t[100000] OK
Time: CPU=0.00408s, GPU Radix=0.01180s, GPU Thrust=0.00027s
Speedup: Radix vs CPU=0.35x, Thrust vs CPU=15.33x, Radix vs Thrust=0.02x

Benchmark: uint64_t[100000] OK
Time: CPU=0.00393s, GPU Radix=0.02357s, GPU Thrust=0.00048s
Speedup: Radix vs CPU=0.17x, Thrust vs CPU=8.20x, Radix vs Thrust=0.02x

Benchmark: int64_t[100000] OK
Time: CPU=0.00427s, GPU Radix=0.02455s, GPU Thrust=0.00035s
Speedup: Radix vs CPU=0.17x, Thrust vs CPU=12.22x, Radix vs Thrust=0.01x

Benchmark: uint32_t[5000000] OK
Time: CPU=0.18671s, GPU Radix=0.03418s, GPU Thrust=0.00084s
Speedup: Radix vs CPU=5.46x, Thrust vs CPU=221.35x, Radix vs Thrust=0.02x

Benchmark: int32_t[5000000] OK
Time: CPU=0.18702s, GPU Radix=0.03589s, GPU Thrust=0.00085s
Speedup: Radix vs CPU=5.21x, Thrust vs CPU=220.51x, Radix vs Thrust=0.02x

Benchmark: uint64_t[5000000] OK
Time: CPU=0.18898s, GPU Radix=0.07209s, GPU Thrust=0.00219s
Speedup: Radix vs CPU=2.62x, Thrust vs CPU=86.26x, Radix vs Thrust=0.03x

Benchmark: int64_t[5000000] OK
Time: CPU=0.19217s, GPU Radix=0.07275s, GPU Thrust=0.00248s
Speedup: Radix vs CPU=2.64x, Thrust vs CPU=77.34x, Radix vs Thrust=0.03x

Benchmark: uint32_t[10000000] OK
Time: CPU=0.36758s, GPU Radix=0.04167s, GPU Thrust=0.00154s
Speedup: Radix vs CPU=8.82x, Thrust vs CPU=238.28x, Radix vs Thrust=0.04x

Benchmark: int32_t[10000000] OK
Time: CPU=0.37177s, GPU Radix=0.04443s, GPU Thrust=0.00155s
Speedup: Radix vs CPU=8.37x, Thrust vs CPU=240.13x, Radix vs Thrust=0.03x

Benchmark: uint64_t[10000000] OK
Time: CPU=0.36685s, GPU Radix=0.09236s, GPU Thrust=0.00407s
Speedup: Radix vs CPU=3.97x, Thrust vs CPU=90.19x, Radix vs Thrust=0.04x

Benchmark: int64_t[10000000] OK
Time: CPU=0.37779s, GPU Radix=0.09465s, GPU Thrust=0.00382s
Speedup: Radix vs CPU=3.99x, Thrust vs CPU=98.89x, Radix vs Thrust=0.04x
```