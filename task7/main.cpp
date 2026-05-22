#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

using namespace std;
using namespace chrono;

const int M = 1024;
const int N = 1024;
const int L = 1024;

// A[M][N], B[N][L], C[M][L]: GEMM
// A[i][k] is at A[i*N + k]
// B[k][j] is at B[k*L + j]
// C[i][j] is at C[i*L + j]

void matmul_ijk(const vector<double> &A, const vector<double> &B,
                vector<double> &C) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < L; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++)
        sum += A[i * N + k] * B[k * L + j];
      C[i * L + j] = sum;
    }
  }
}

void matmul_ikj(const vector<double> &A, const vector<double> &B,
                vector<double> &C) {
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < N; k++) {
      double aik = A[i * N + k];
      for (int j = 0; j < L; j++)
        C[i * L + j] += aik * B[k * L + j];
    }
  }
}

void matmul_ikj_omp(const vector<double> &A, const vector<double> &B,
                    vector<double> &C) {
#pragma omp parallel for
  for (int i = 0; i < M; i++) {
    for (int k = 0; k < N; k++) {
      double aik = A[i * N + k];
      for (int j = 0; j < L; j++)
        C[i * L + j] += aik * B[k * L + j];
    }
  }
}

void matmul_jki(const vector<double> &A, const vector<double> &B,
                vector<double> &C) {
  for (int j = 0; j < L; j++) {
    for (int k = 0; k < N; k++) {
      double bkj = B[k * L + j];
      for (int i = 0; i < M; i++)
        C[i * L + j] += A[i * N + k] * bkj;
    }
  }
}

int main() {
  printf("GEMM loop-order comparison: A[%d][%d] x B[%d][%d] = C[%d][%d]\n", M,
         N, N, L, M, L);

  vector<double> A(M * N), B(N * L), C(M * L);

  for (int i = 0; i < M * N; i++)
    A[i] = rand() % 100 / 10.0;
  for (int i = 0; i < N * L; i++)
    B[i] = rand() % 100 / 10.0;

  // warmup 5 times with ijk
  for (int rep = 0; rep < 5; rep++) {
    fill(C.begin(), C.end(), 0.0);
    matmul_ijk(A, B, C);
  }

  // reference
  double ref_trace = 0.0;
  for (int i = 0; i < min(M, L); i++)
    ref_trace += C[i * L + i];

  using Clock = high_resolution_clock;

  const int REPTIME = 20;

  double t_ijk = 0;
  for (int rep = 0; rep < REPTIME; rep++) {
    fill(C.begin(), C.end(), 0.0);
    auto t0 = Clock::now();
    matmul_ijk(A, B, C);
    auto t1 = Clock::now();
    t_ijk += duration<double>(t1 - t0).count();
  }
  double trace_ijk = 0.0;
  for (int i = 0; i < min(M, L); i++)
    trace_ijk += C[i * L + i];

  double t_ikj = 0;
  for (int rep = 0; rep < REPTIME; rep++) {
    fill(C.begin(), C.end(), 0.0);
    auto t0 = Clock::now();
    matmul_ikj(A, B, C);
    auto t1 = Clock::now();
    t_ikj += duration<double>(t1 - t0).count();
  }
  double trace_ikj = 0.0;
  for (int i = 0; i < min(M, L); i++)
    trace_ikj += C[i * L + i];

  double t_jki = 0;
  for (int rep = 0; rep < REPTIME; rep++) {
    fill(C.begin(), C.end(), 0.0);
    auto t0 = Clock::now();
    matmul_jki(A, B, C);
    auto t1 = Clock::now();
    t_jki += duration<double>(t1 - t0).count();
  }
  double trace_jki = 0.0;
  for (int i = 0; i < min(M, L); i++)
    trace_jki += C[i * L + i];

  double t_ikj_omp = 0;
  for (int rep = 0; rep < REPTIME; rep++) {
    fill(C.begin(), C.end(), 0.0);
    auto t0 = Clock::now();
    matmul_ikj_omp(A, B, C);
    auto t1 = Clock::now();
    t_ikj_omp += duration<double>(t1 - t0).count();
  }
  double trace_ikj_omp = 0.0;
  for (int i = 0; i < min(M, L); i++)
    trace_ikj_omp += C[i * L + i];

  double flops = 2.0 * REPTIME * M * N * L;

  printf("%-10s  %12s  %12s  %12s  %14s\n", "version", "time(s)", "MFLOPS",
         "SPEEDUP", "tr(C) check");
  printf("%-10s  %12.4f  %12.2f  %12.2f  %14s\n", "ijk", t_ijk,
         flops / t_ijk / 1e6, t_ijk / t_ijk,
         abs(trace_ijk - ref_trace) < 1e-6 ? "OK" : "FAIL");
  printf("%-10s  %12.4f  %12.2f  %12.2f  %14s\n", "ikj", t_ikj,
         flops / t_ikj / 1e6, t_ijk / t_ikj,
         abs(trace_ikj - ref_trace) < 1e-6 ? "OK" : "FAIL");
  printf("%-10s  %12.4f  %12.2f  %12.2f  %14s\n", "jki", t_jki,
         flops / t_jki / 1e6, t_ijk / t_jki,
         abs(trace_jki - ref_trace) < 1e-6 ? "OK" : "FAIL");
  printf("%-10s  %12.4f  %12.2f  %12.2f  %14s\n", "ikj+omp", t_ikj_omp,
         flops / t_ikj_omp / 1e6, t_ijk / t_ikj_omp,
         abs(trace_ikj_omp - ref_trace) < 1e-6 ? "OK" : "FAIL");
  return 0;
}
