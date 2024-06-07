# General matrix multiplication

## Introducation
This repository contains a simple implementation of matrix multiplication using OpenMP and the NEON instruction set. The goal is to demonstrate the use of parallel processing and optimized instructions for matrix operations.

## Requirements
1. A compatible ARM-based processor with NEON support
2. OpenMP installed and configured on your system
3. A C compiler (e.g., GCC/CLANG)

### Compiliation
To compile the code, use the following command:
```bash
 gcc -o gemm gemm.c -O3 -ffast-math -fopenmp -march=native
```

## Matrix Multiplication Benchmark on M2 Pro Processor
This benchmark compares the performance of four different matrix multiplication implementations on an M2 Pro processor.
Implementations:
1. Optimized Neon Parallel BLOCKED
2. Standard Neon Parallel BLOCKED
3. Normal Parallel NEON
4. Normal Parallel matmul

### Results:
1. N = 1024, BLOCK_SIZE = 16
  - Optimized: 87.17 GFLOP/S 
  - Standard: 69.49 GFLOP/S 
  - Normal NEON: 76.44 GFLOP/S 
  - Normal matmul: 4.85 GFLOP/S s
2. N = 8192, BLOCK_SIZE = 16
  - Optimized: 122.27 GFLOP/S ms
  - Standard: 72.04 GFLOP/S  ms
  - Normal NEON: 60.23 GFLOP/S  ms
  - Normal matmul: Not applicable

> [!NOTE] 
> Our old best performance was ~76 GFLOP/S, and we have now achieved a  significant improvement of 122.27 GFLOP/S with our optimized implementation.