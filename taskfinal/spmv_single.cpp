#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "csr.h"

using namespace std;

int main(int argc, char *argv[]) {
  int nthreads = 4;
  int grid_size = 2000;
  int iters = 100;
  if (argc > 1) nthreads = atoi(argv[1]);
  if (argc > 2) grid_size = atoi(argv[2]);
  if (argc > 3) iters = atoi(argv[3]);

  omp_set_num_threads(nthreads);

  int N = grid_size * grid_size;
  CSR A = generate_laplacian(grid_size);

  vector<float> xv(N), yv(N);
  float *x = xv.data(), *y = yv.data();
  for (int i = 0; i < N; i++)
    x[i] = (i % 100) / 10.0f;

  // warmup
  for (int r = 0; r < 10; r++)
    spmv_omp(A, x, y);

  // measured SpMV calls
  for (int r = 0; r < iters; r++)
    spmv_omp(A, x, y);

  printf("y[0]=%f\n", y[0]);
  return 0;
}
