#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "csr.h"

using namespace std;
using namespace chrono;
using Clock = high_resolution_clock;

double bench_serial(const CSR &A, const float *x, float *y, int reptime) {
  double total = 0;
  for (int rep = 0; rep < reptime; rep++) {
    auto t0 = Clock::now();
    spmv(A, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  return total / reptime;
}

double bench_omp(const CSR &A, const float *x, float *y, int reptime) {
  double total = 0;
  for (int rep = 0; rep < reptime; rep++) {
    auto t0 = Clock::now();
    spmv_omp(A, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  return total / reptime;
}

double bench_ell_serial(const ELL &E, const float *x, float *y, int reptime) {
  double total = 0;
  for (int rep = 0; rep < reptime; rep++) {
    auto t0 = Clock::now();
    spmv_ell(E, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  return total / reptime;
}

double bench_ell_omp(const ELL &E, const float *x, float *y, int reptime) {
  double total = 0;
  for (int rep = 0; rep < reptime; rep++) {
    auto t0 = Clock::now();
    spmv_ell_omp(E, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  return total / reptime;
}

void run_bench(const char *name, CSR &A, int N, int reptime) {
  vector<float> xv(N), yv(N), y_refv(N);
  float *x = xv.data(), *y = yv.data(), *y_ref = y_refv.data();

  for (int i = 0; i < N; i++)
    x[i] = (i % 100) / 10.0f;

  // CSR benchmark
  spmv(A, x, y_ref);
  double t_s = bench_serial(A, x, y, reptime);
  memcpy(y_ref, y, N * sizeof(float));
  double t_o = bench_omp(A, x, y, reptime);

  // ELLPACK benchmark
  ELL E = csr_to_ell(A);
  double t_ell_s = bench_ell_serial(E, x, y, reptime);
  double t_ell_o = bench_ell_omp(E, x, y, reptime);

  // verify ELL correctness
  vector<float> y_ell(N);
  spmv_ell(E, x, y_ell.data());
  float err = 0;
  for (int i = 0; i < N; i++) err = max(err, fabsf(y_ell[i] - y_ref[i]));

  printf("%-16s  %7d  %10d  %3d  %8.4f  %8.4f  %5.1fx  %8.4f  %8.4f  %5.1fx  %s\n",
         name, N, A.nnz, E.max_nnz,
         t_s, t_o, t_s / t_o,
         t_ell_s, t_ell_o, t_ell_s / t_ell_o,
         err < 1e-5f ? "OK" : "FAIL");
}

int main(int argc, char *argv[]) {
  int nthreads = 4;
  if (argc > 1) nthreads = atoi(argv[1]);

  const int REPTIME = 500;
  omp_set_num_threads(nthreads);

  printf("%-16s  %7s  %10s  %3s  %8s  %8s  %5s  %8s  %8s  %5s  %s\n",
         "matrix", "N", "nnz", "max",
         "csr(s)", "csr_omp", "speed", "ell(s)", "ell_omp", "speed", "check");
  printf("--------------------------------------------------------------------------------------------------------------\n");

  {
    int gs = 2000, N = gs * gs;
    CSR A = generate_laplacian(gs);
    run_bench("Laplacian", A, N, REPTIME);
  }

  {
    int N = 2000000;
    CSR A = generate_random_banded(N, 100, 10);
    run_bench("Banded-narrow", A, N, REPTIME);
  }

  {
    int N = 1000000;
    CSR A = generate_random_banded(N, 2000, 50);
    run_bench("Banded-wide", A, N, REPTIME);
  }

  {
    int N = 3000;
    CSR A = generate_erdos_renyi(N, 0.005);
    run_bench("ErdosRenyi", A, N, REPTIME);
  }

  {
    int N = 2000000;
    CSR A = generate_power_law(N, 10);
    run_bench("PowerLaw", A, N, REPTIME);
  }

  return 0;
}
