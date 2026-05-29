/*
  A = LU
*/
#include <math.h>
#include <omp.h>
#include <stdio.h>

#define N 1024

void matmul(double *pa, double *pb, double *pc);
double errmat(double *pa, double *pb);
void copymat(double *pa, double *pb);

int main() {
  int i, j, k;
  static double A[N][N], Atmp[N][N];
  double t;

  /* set initial value */
  for (i = 0; i < N; i++)
    for (j = 0; j < N; j++)
      A[i][j] = 0.0;
  for (i = 1; i < N - 1; i++) {
    A[i][i] = 2.0;
    A[i][i - 1] = -1.0;
    A[i][i + 1] = -1.0;
  }
  A[0][0] = 2.0;
  A[0][1] = -1.0;
  A[N - 1][N - 1] = 2.0;
  A[N - 1][N - 2] = -1.0;
  copymat(A[0], Atmp[0]);

  printf("N = %d, threads = %d\n", N, omp_get_max_threads());

  // i-k-j changed to k-i-j
  t = omp_get_wtime();
  for (k = 0; k < N; k++) {
#pragma omp parallel for schedule(static)
    for (i = k + 1; i < N; i++)
      A[i][k] /= A[k][k];

#pragma omp parallel for collapse(2) schedule(static)
    for (i = k + 1; i < N; i++)
      for (j = k + 1; j < N; j++)
        A[i][j] -= A[i][k] * A[k][j];
  }
  t = omp_get_wtime() - t;
  printf("elapsed time %.2e, ", t);

  // for checking
  static double L[N][N], U[N][N], B[N][N];
  double err = 0.0;

  for (i = 0; i < N; i++)
    for (j = 0; j < i; j++)
      L[i][j] = A[i][j];
  for (i = 0; i < N; i++)
    L[i][i] = 1.0;
  for (i = 0; i < N; i++)
    for (j = i; j < N; j++)
      U[i][j] = A[i][j];

  matmul(L[0], U[0], B[0]);
  err = errmat(Atmp[0], B[0]);
  printf("err= %.2e\n", err);

  return 0;
}

void matmul(double *pa, double *pb, double *pc) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      double sum = 0.0;
      for (int k = 0; k < N; k++)
        sum += *(pa + i * N + k) * *(pb + k * N + j);
      *(pc + i * N + j) = sum;
    }
}

double errmat(double *pa, double *pb) {
  double err = 0.0;

#pragma omp parallel for reduction(+ : err) schedule(static)
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      err += fabs(*(pb + i * N + j) - *(pa + i * N + j));

  return err;
}

void copymat(double *pa, double *pb) {
#pragma omp parallel for schedule(static)
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
      *(pb + i * N + j) = *(pa + i * N + j);
}