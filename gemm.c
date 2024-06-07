#include <arm_neon.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 2048
#define BLOCK_SIZE 16

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
float *A;
float *B;
float *BT;
float *C;
float *val;
float *val2;
float *val3;

int main() {
  // Dynamic Creation with memory alignment 
  posix_memalign((void **)&A, 64, N * N * sizeof(float));
  posix_memalign((void **)&B, 64, N * N * sizeof(float));
  posix_memalign((void **)&BT, 64, N * N * sizeof(float));
  posix_memalign((void **)&C, 64, N * N * sizeof(float));
  posix_memalign((void **)&val, 64, N * N * sizeof(float));
  posix_memalign((void **)&val2, 64, N * N * sizeof(float));
  posix_memalign((void **)&val3, 64, N * N * sizeof(float));

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
          float32x4_t sum0 = vdupq_n_f32(0.0);
          float32x4_t sum1 = vdupq_n_f32(0.0);
          float32x4_t sum2 = vdupq_n_f32(0.0);
          float32x4_t sum3 = vdupq_n_f32(0.0);

          __builtin_prefetch(&A[kk * N], 0, 3);
          __builtin_prefetch(&BT[jj * N], 0, 3);

          for (k = 0; k < N; k += 16) {
            float32x4x4_t va = vld4q_f32(&A[kk * N + k]);
            float32x4x4_t vb = vld4q_f32(&BT[jj * N + k]);

            sum0 = vmlaq_f32(sum0, va.val[0], vb.val[0]);
            sum1 = vmlaq_f32(sum1, va.val[1], vb.val[1]);
            sum2 = vmlaq_f32(sum2, va.val[2], vb.val[2]);
            sum3 = vmlaq_f32(sum3, va.val[3], vb.val[3]);
          }

          float32x4_t sum =
              vaddq_f32(vaddq_f32(sum0, sum1), vaddq_f32(sum2, sum3));
          val3[jj * N + kk] = vaddvq_f32(sum);
        }
      }
    }
  }
  double gflop = (2.0 * N * N * N) * 1e-9;
  uint64_t end = nanos();
  double s = (end - start) * 1e-9;
  printf("Optimized Neon Parallel BLOCKED Matrix Multiplication (N = %d, "
         "BLOCK_SIZE = %d) achieves %.2f GFLOP/S in %.2f ms\n",
         N, BLOCK_SIZE, gflop / s, s * 1e3);

  start = nanos();
  i = 0, j = 0, k = 0, kk = 0, jj = 0;
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
  end = nanos();
  s = (end - start) * 1e-9;
  printf("Standard Neon Parallel BLOCKED Matrix Multiplication (N = %d, BLOCK_SIZE = %d) takes %.2f ms, achieving a performance of %.2f GFLOP/S\n",
         N, BLOCK_SIZE, s * 1e3, gflop / s);

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
 printf("Normal Parallel NEON Matrix Multiplication (N = %d) takes %.2f ms, achieving a performance of %.2f GFLOP/S\n",
         N, s * 1e3, gflop / s);

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
  printf("Normal Parallel matmul gets: %f GFLOP/S -- %.2f ms\n", gflop / s,
         s * 1e3);

  printf("\nThe difference is : %d", check_equality(C, val3));

  free(A);
  free(B);
  free(BT);
  free(C);
  free(val);
  free(val2);
  free(val3);
  return 0;
}