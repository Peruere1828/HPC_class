#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "csr.h"
#include "csc.h"
#include "mmio.h"

using namespace std;
using namespace chrono;
using Clock = high_resolution_clock;

double bench_csc_serial(const CSC &A, const float *x, float *y, int reptime) {
  double total = 0;
  for (int rep = 0; rep < reptime; rep++) {
    auto t0 = Clock::now();
    spmv_csc(A, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  return total / reptime;
}

double bench_csc_omp(const CSC &A, const float *x, float *y, int reptime) {
  double total = 0;
  for (int rep = 0; rep < reptime; rep++) {
    auto t0 = Clock::now();
    spmv_csc_omp(A, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  return total / reptime;
}

void run_bench(const char *name, CSR &A_csr) {
  // Convert CSR to CSC
  CSC A = convert_csr_to_csc(A_csr);

  int N = A.n;
  int M = A.m;
  int reptime = N > 500000 ? 50 : N > 100000 ? 200 : 500;
  vector<float> xv(M), yv(N), y_refv(N);
  float *x = xv.data(), *y = yv.data(), *y_ref = y_refv.data();

  for (int i = 0; i < M; i++)
    x[i] = (i % 100) / 10.0f;

  // SpMV: each non-zero = 1 multiply + 1 add = 2 FLOPs
  double flops_per_spmv = 2.0 * A.nnz;

  // CSC serial benchmark
  spmv_csc(A, x, y_ref);
  double t_s = bench_csc_serial(A, x, y, reptime);

  // CSC OpenMP benchmark
  double t_o = bench_csc_omp(A, x, y, reptime);

  // Verify OMP result against serial (use relative error)
  float max_err = 0.0f, max_ref = 0.0f;
  for (int i = 0; i < N; i++) {
    max_err = max(max_err, fabsf(y[i] - y_ref[i]));
    max_ref = max(max_ref, fabsf(y_ref[i]));
  }
  float rel_err = max_err / max(1.0f, max_ref);

  double gflops_s = flops_per_spmv / (t_s * 1e9);
  double gflops_o = flops_per_spmv / (t_o * 1e9);

  printf("%-20s  %7d  %10d  %8.4f  %8.4f  %8.2f  %8.2f  %5.1fx  %s\n",
         name, N, A.nnz,
         t_s, t_o, gflops_s, gflops_o, t_s / t_o,
         rel_err < 1e-5f ? "OK" : "FAIL");
}

void print_header() {
  printf("%-20s  %7s  %10s  %8s  %8s  %8s  %8s  %5s  %4s\n",
         "matrix", "N", "nnz", "csc(s)", "csc_omp", "csc_gflops", "omp_gflops", "speed", "chk");
  printf("------------------------------------------------------------------------------------------\n");
}

int main(int argc, char *argv[]) {
  int nthreads = 4;
  if (argc > 1) nthreads = atoi(argv[1]);

  omp_set_num_threads(nthreads);

  print_header();

  // Load .mtx files from command line: ./spmv_csc 4 file1.mtx file2.mtx ...
  for (int i = 2; i < argc; i++) {
    CSR A = load_mm(argv[i]);
    const char *name = argv[i];
    const char *slash = strrchr(name, '/');
    if (slash) name = slash + 1;
    char buf[256];
    strncpy(buf, name, sizeof(buf));
    char *dot = strrchr(buf, '.');
    if (dot) *dot = '\0';
    run_bench(buf, A);
  }

  // If no mtx files given, run built-in generators
  if (argc <= 2) {
    {
      int gs = 2000, N = gs * gs;
      CSR A = generate_laplacian(gs);
      run_bench("Laplacian", A);
    }
    {
      int N = 2000000;
      CSR A = generate_random_banded(N, 100, 10);
      run_bench("Banded-narrow", A);
    }
    {
      int N = 1000000;
      CSR A = generate_random_banded(N, 2000, 50);
      run_bench("Banded-wide", A);
    }
    {
      int N = 3000;
      CSR A = generate_erdos_renyi(N, 0.005);
      run_bench("ErdosRenyi", A);
    }
    {
      int N = 2000000;
      CSR A = generate_power_law(N, 10);
      run_bench("PowerLaw", A);
    }
  }

  return 0;
}
