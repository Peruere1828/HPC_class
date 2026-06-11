#ifndef CSR_H
#define CSR_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace std;

// 64-byte aligned allocation (cache line size)
static inline float *alloc_float(size_t n) {
  size_t bytes = n * sizeof(float);
  if (bytes % 64) bytes = ((bytes / 64) + 1) * 64;
  return (float *)aligned_alloc(64, bytes);
}
static inline int *alloc_int(size_t n) {
  size_t bytes = n * sizeof(int);
  if (bytes % 64) bytes = ((bytes / 64) + 1) * 64;
  return (int *)aligned_alloc(64, bytes);
}

struct CSR {
  int n;    // rows
  int m;    // cols
  int nnz;
  float *val;
  int *col_idx;
  int *row_ptr;

  CSR() : n(0), m(0), nnz(0), val(nullptr), col_idx(nullptr), row_ptr(nullptr) {}

  CSR(CSR &&o) noexcept
      : n(o.n), m(o.m), nnz(o.nnz), val(o.val), col_idx(o.col_idx), row_ptr(o.row_ptr) {
    o.val = nullptr;
    o.col_idx = nullptr;
    o.row_ptr = nullptr;
  }
  CSR &operator=(CSR &&o) noexcept {
    if (this != &o) {
      free(val);
      free(col_idx);
      free(row_ptr);
      n = o.n;
      m = o.m;
      nnz = o.nnz;
      val = o.val;
      col_idx = o.col_idx;
      row_ptr = o.row_ptr;
      o.val = nullptr;
      o.col_idx = nullptr;
      o.row_ptr = nullptr;
    }
    return *this;
  }
  ~CSR() {
    free(val);
    free(col_idx);
    free(row_ptr);
  }
  CSR(const CSR &) = delete;
  CSR &operator=(const CSR &) = delete;
};

// 2D Laplacian 五点差分: grid_size×grid_size 网格 → N×N 矩阵, N=grid_size^2
CSR generate_laplacian(int grid_size) {
  int N = grid_size * grid_size;
  CSR A;
  A.n = N;
  A.m = N;
  A.row_ptr = alloc_int(N + 1);
  memset(A.row_ptr, 0, (N + 1) * sizeof(int));

  for (int i = 0; i < N; i++) {
    int r = i / grid_size, c = i % grid_size;
    int cnt = 1;
    if (r > 0) cnt++;
    if (r < grid_size - 1) cnt++;
    if (c > 0) cnt++;
    if (c < grid_size - 1) cnt++;
    A.row_ptr[i + 1] = A.row_ptr[i] + cnt;
  }

  A.nnz = A.row_ptr[N];
  A.val = alloc_float(A.nnz);
  A.col_idx = alloc_int(A.nnz);

  int idx = 0;
  for (int i = 0; i < N; i++) {
    int r = i / grid_size, c = i % grid_size;
    if (r > 0)             { A.col_idx[idx] = i - grid_size; A.val[idx] = -1.0f; idx++; }
    if (c > 0)             { A.col_idx[idx] = i - 1;         A.val[idx] = -1.0f; idx++; }
    {                        A.col_idx[idx] = i;              A.val[idx] =  4.0f; idx++; }
    if (c < grid_size - 1) { A.col_idx[idx] = i + 1;         A.val[idx] = -1.0f; idx++; }
    if (r < grid_size - 1) { A.col_idx[idx] = i + grid_size; A.val[idx] = -1.0f; idx++; }
  }

  return A;
}

// Random sparse matrix: 每行有固定数量的随机非零元 (bandwidth-limited)
CSR generate_random_banded(int N, int band_width, int nnz_per_row) {
  CSR A;
  A.n = N;
  A.m = N;
  A.row_ptr = alloc_int(N + 1);

  nnz_per_row = min(nnz_per_row, band_width);

  A.row_ptr[0] = 0;
  for (int i = 0; i < N; i++)
    A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row;

  A.nnz = A.row_ptr[N];
  A.val = alloc_float(A.nnz);
  A.col_idx = alloc_int(A.nnz);

  unsigned seed = 42;
  int idx = 0;
  for (int i = 0; i < N; i++) {
    int lo = max(0, i - band_width / 2);
    int hi = min(N, lo + band_width);
    int cnt = min(nnz_per_row, hi - lo);

    for (int k = 0; k < cnt; k++) {
      seed = seed * 1103515245 + 12345;
      int col = lo + (int)((unsigned)seed % (hi - lo));
      A.col_idx[idx] = col;
      A.val[idx] = 1.0f + (float)((unsigned)seed % 1000) / 1000.0f;
      idx++;
    }
  }
  A.nnz = idx;
  A.row_ptr[N] = idx;

  return A;
}

// Erdos-Renyi random sparse: 每个元素以概率 p 独立出现
CSR generate_erdos_renyi(int N, double p) {
  CSR A;
  A.n = N;
  A.m = N;
  A.row_ptr = alloc_int(N + 1);

  unsigned seed = 12345;
  vector<int> row_nnz(N, 0);
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      seed = seed * 1103515245 + 12345;
      if ((double)((unsigned)seed % 10000) / 10000.0 < p)
        row_nnz[i]++;
    }
  }

  A.row_ptr[0] = 0;
  for (int i = 0; i < N; i++)
    A.row_ptr[i + 1] = A.row_ptr[i] + row_nnz[i];

  A.nnz = A.row_ptr[N];
  A.val = alloc_float(A.nnz);
  A.col_idx = alloc_int(A.nnz);

  int idx = 0;
  seed = 12345;
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      seed = seed * 1103515245 + 12345;
      if ((double)((unsigned)seed % 10000) / 10000.0 < p) {
        A.col_idx[idx] = j;
        A.val[idx] = 1.0f + (float)((unsigned)seed % 1000) / 1000.0f;
        idx++;
      }
    }
  }

  return A;
}

// Power-law graph adjacency matrix
CSR generate_power_law(int N, int nnz_per_row) {
  CSR A;
  A.n = N;
  A.m = N;
  A.row_ptr = alloc_int(N + 1);

  A.row_ptr[0] = 0;
  for (int i = 0; i < N; i++)
    A.row_ptr[i + 1] = A.row_ptr[i] + nnz_per_row;

  A.nnz = A.row_ptr[N];
  A.val = alloc_float(A.nnz);
  A.col_idx = alloc_int(A.nnz);

  unsigned seed = 99999;
  int idx = 0;
  for (int i = 0; i < N; i++) {
    for (int k = 0; k < nnz_per_row; k++) {
      seed = seed * 1103515245 + 12345;
      double u = (double)((unsigned)seed % 100000) / 100000.0;
      int col = (int)(u * u * N);
      if (col >= N) col = N - 1;
      if (col == i) col = (col + 1) % N;
      A.col_idx[idx] = col;
      A.val[idx] = 1.0f + (float)((unsigned)seed % 1000) / 1000.0f;
      idx++;
    }
  }

  return A;
}

// CSR SpMV: y = A * x
void spmv(const CSR &A, const float *x, float *y) {
  const int *rp = A.row_ptr;
  const int *ci = A.col_idx;
  const float *vl = A.val;

  for (int i = 0; i < A.n; i++) {
    float sum = 0.0f;
    for (int j = rp[i]; j < rp[i + 1]; j++)
      sum += vl[j] * x[ci[j]];
    y[i] = sum;
  }
}

// CSR SpMV OpenMP
void spmv_omp(const CSR &A, const float *x, float *y) {
  const int *rp = A.row_ptr;
  const int *ci = A.col_idx;
  const float *vl = A.val;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < A.n; i++) {
    float sum = 0.0f;
    for (int j = rp[i]; j < rp[i + 1]; j++)
      sum += vl[j] * x[ci[j]];
    y[i] = sum;
  }
}

#endif
