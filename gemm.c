#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 8192
#define BLOCK_SIZE 32

float A[N * N] __attribute__((aligned(64)));
float B[N * N] __attribute__((aligned(64)));
float BT[N * N] __attribute__((aligned(64)));
float C[N * N] __attribute__((aligned(64)));
float val[N * N] __attribute__((aligned(64)));
float val2[N * N] __attribute__((aligned(64)));

int check_equality(float *matrix, float *matrix2) {
  float diff = 0;
  for (int i = 0; i < N * N; i++) {
    diff += fabs(matrix[i] - matrix2[i]);
  }
  return diff;
}

void print_matrix(float *matrix) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {

      printf("%.0f, ", matrix[i * N + j]);
    }
    printf("\n");
  }
}

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main() {
  for (int i = 0; i < N * N; i++) {
    A[i] = (float)(rand() % 100);
    B[i] = (float)(rand() % 100);
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      BT[i * N + j] = B[j * N + i];
    }
  }

  uint64_t start = nanos();
  int i, j, k, kk, jj;
#pragma omp parallel for private(i, j, k, kk, jj) shared(A, BT, val)
  for (i = 0; i < N; i += BLOCK_SIZE) {
    for (j = 0; j < N; j += BLOCK_SIZE) {
      for (kk = i; kk < i + BLOCK_SIZE && kk < N; ++kk) {
        for (jj = j; jj < j + BLOCK_SIZE && jj < N; ++jj) {
          float32x4_t sum = vdupq_n_f32(0.0);

          __builtin_prefetch(&A[kk * N], 0, 3);
          __builtin_prefetch(&BT[jj * N], 0, 3);

          for (k = 0; k < N; k += 4) {
            float32x4_t va = vld1q_f32(&A[kk * N + k]);
            float32x4_t vb = vld1q_f32(&BT[jj * N + k]);
            sum = vmlaq_f32(sum, va, vb);
          }
          val[jj * N + kk] = vaddvq_f32(sum);
        }
      }
    }
  }
  uint64_t end = nanos();
  double gflop = (2.0 * N * N * N) * 1e-9;
  double s = (end - start) * 1e-9;
  printf("Neon Parallel BLOCKED matmul gets: %f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);

  start = nanos();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      float32x4_t sum = {0.0, 0.0, 0.0, 0.0};
      __builtin_prefetch(&A[i * N], 0, 3);
      __builtin_prefetch(&BT[j * N], 0, 3);

      for (int k = 0; k < N; k += 4) {
        float32x4_t va = vld1q_f32(&A[i * N + k]);
        float32x4_t vb = vld1q_f32(&BT[j * N + k]);
        sum = vmlaq_f32(sum, va, vb);
      }
      // we could handle N not multiple of 4 but why ?
      val2[j * N + i] = vaddvq_f32(sum);
    }
  }
  end = nanos();
  s = (end - start) * 1e-9;
  printf("Normal Parallel NEON matmul gets: %f GFLOP/S -- %.2f ms\n", gflop / s,
         s * 1e3);

  start = nanos();
#pragma omp parallel for collapse(2)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      int sum = 0;
      for (int k = 0; k < N; k++) {
        sum += A[i * N + k] * B[k * N + j];
      }
      C[j * N + i] = sum;
    }
  }
  end = nanos();
  s = (end - start) * 1e-9;
  printf("Normal Parallel matmul gets: %f GFLOP/S -- %.2f ms\n", gflop / s, s * 1e3);

  printf("\nThe difference is : %d", check_equality(C, val2));
  return 0;
}