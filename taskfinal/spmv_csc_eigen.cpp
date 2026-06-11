#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>
#include <algorithm>

#include <Eigen/SparseCore>
#include "csr.h"
#include "csc.h"
#include "mmio.h"

using namespace std;
using namespace chrono;
using Clock = high_resolution_clock;

void run_bench(const char *name, CSR &A_csr, int nthreads) {
  int N = A_csr.n, M = A_csr.m;
  int reptime = N > 500000 ? 50 : N > 100000 ? 200 : 500;

  // Convert CSR to CSC
  CSC A_csc = convert_csr_to_csc(A_csr);

  // Build Eigen sparse matrix in ColMajor (CSC) format
  Eigen::SparseMatrix<float, Eigen::ColMajor> Em(N, M);
  {
    vector<Eigen::Triplet<float>> trips;
    trips.reserve(A_csr.nnz);
    for (int i = 0; i < N; i++)
      for (int j = A_csr.row_ptr[i]; j < A_csr.row_ptr[i+1]; j++)
        trips.emplace_back(i, A_csr.col_idx[j], A_csr.val[j]);
    Em.setFromTriplets(trips.begin(), trips.end());
  }

  vector<float> xv(M), yv(N);
  for (int i = 0; i < M; i++) xv[i] = (i % 100) / 10.0f;
  float *x = xv.data(), *y = yv.data();

  Eigen::Map<Eigen::VectorXf> ex(x, M);
  Eigen::VectorXf ey(N);

  // Warmup
  spmv_csc(A_csc, x, y);
  spmv_csc_omp(A_csc, x, y);
  ey = Em * ex;

  // Bench CSC serial
  double total = 0;
  for (int r = 0; r < reptime; r++) {
    auto t0 = Clock::now();
    spmv_csc(A_csc, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  double t_csc = total / reptime;

  // Bench CSC OpenMP
  total = 0;
  for (int r = 0; r < reptime; r++) {
    auto t0 = Clock::now();
    spmv_csc_omp(A_csc, x, y);
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  double t_omp = total / reptime;

  // Bench Eigen (ColMajor = CSC)
  total = 0;
  for (int r = 0; r < reptime; r++) {
    auto t0 = Clock::now();
    ey = Em * ex;
    auto t1 = Clock::now();
    total += duration<double>(t1 - t0).count();
  }
  double t_eig = total / reptime;

  // Verify against CSC serial
  spmv_csc(A_csc, x, y);
  float err = 0;
  for (int i = 0; i < N; i++)
    err = max(err, fabsf(y[i] - ey(i)));

  // SpMV: each non-zero = 1 multiply + 1 add = 2 FLOPs
  double flops_per_spmv = 2.0 * A_csr.nnz;
  double gflops_csc  = flops_per_spmv / (t_csc * 1e9);
  double gflops_omp  = flops_per_spmv / (t_omp * 1e9);
  double gflops_eig  = flops_per_spmv / (t_eig * 1e9);

  printf("%-20s  %7d  %10d  %8.4f  %8.4f  %8.4f  %7.2f  %7.2f  %7.2f  %5.1fx  %5.1fx  %s\n",
         name, N, A_csr.nnz, t_csc * 1000, t_omp * 1000, t_eig * 1000,
         gflops_csc, gflops_omp, gflops_eig,
         t_csc / t_omp, t_csc / t_eig,
         err < 1e-3f ? "OK" : "FAIL");
}

int main(int argc, char *argv[]) {
  int nthreads = 4;
  if (argc > 1) nthreads = atoi(argv[1]);

  omp_set_num_threads(nthreads);

  printf("%-20s  %7s  %10s  %8s  %8s  %8s  %7s  %7s  %7s  %5s  %5s  %s\n",
         "matrix", "N", "nnz", "csc(ms)", "omp(ms)", "eig(ms)",
         "csc_gf", "omp_gf", "eig_gf", "omp↑", "eig↑", "chk");
  printf("----------------------------------------------------------------------------------------------------\n");

  // Load .mtx files
  for (int i = 2; i < argc; i++) {
    CSR A = load_mm(argv[i]);
    const char *name = argv[i];
    const char *slash = strrchr(name, '/');
    if (slash) name = slash + 1;
    char buf[256];
    strncpy(buf, name, sizeof(buf));
    char *dot = strrchr(buf, '.');
    if (dot) *dot = '\0';
    run_bench(buf, A, nthreads);
  }

  // Built-in generators if no mtx files
  if (argc <= 2) {
    {
      int gs = 2000, N = gs * gs;
      CSR A = generate_laplacian(gs);
      run_bench("Laplacian", A, nthreads);
    }
    {
      int N = 2000000;
      CSR A = generate_random_banded(N, 100, 10);
      run_bench("Banded-narrow", A, nthreads);
    }
    {
      int N = 1000000;
      CSR A = generate_random_banded(N, 2000, 50);
      run_bench("Banded-wide", A, nthreads);
    }
    {
      int N = 3000;
      CSR A = generate_erdos_renyi(N, 0.005);
      run_bench("ErdosRenyi", A, nthreads);
    }
    {
      int N = 2000000;
      CSR A = generate_power_law(N, 10);
      run_bench("PowerLaw", A, nthreads);
    }
  }

  return 0;
}
