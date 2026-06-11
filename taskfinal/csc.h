#ifndef CSC_H
#define CSC_H

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "csr.h"

using namespace std;

// CSC: Compressed Sparse Column format
//   col_ptr[j]   = start index in val[]/row_idx[] for column j
//   row_idx[k]   = row index of the k-th non-zero
//   val[k]       = value of the k-th non-zero
struct CSC {
  int n;    // rows
  int m;    // cols
  int nnz;  // number of non-zeros
  float *val;
  int *row_idx;
  int *col_ptr;

  CSC() : n(0), m(0), nnz(0), val(nullptr), row_idx(nullptr), col_ptr(nullptr) {}

  CSC(CSC &&o) noexcept
      : n(o.n), m(o.m), nnz(o.nnz), val(o.val), row_idx(o.row_idx), col_ptr(o.col_ptr) {
    o.val = nullptr;
    o.row_idx = nullptr;
    o.col_ptr = nullptr;
  }
  CSC &operator=(CSC &&o) noexcept {
    if (this != &o) {
      free(val);
      free(row_idx);
      free(col_ptr);
      n = o.n;
      m = o.m;
      nnz = o.nnz;
      val = o.val;
      row_idx = o.row_idx;
      col_ptr = o.col_ptr;
      o.val = nullptr;
      o.row_idx = nullptr;
      o.col_ptr = nullptr;
    }
    return *this;
  }
  ~CSC() {
    free(val);
    free(row_idx);
    free(col_ptr);
  }
  CSC(const CSC &) = delete;
  CSC &operator=(const CSC &) = delete;
};

// Convert CSR to CSC (transpose storage format behind the same matrix)
// CSR: row-major traversal → CSC: column-major traversal
inline CSC convert_csr_to_csc(const CSR &A) {
  CSC B;
  B.n = A.n;
  B.m = A.m;
  B.nnz = A.nnz;
  B.col_ptr = alloc_int(B.m + 1);
  B.row_idx = alloc_int(B.nnz);
  B.val = alloc_float(B.nnz);

  // Count non-zeros per column
  memset(B.col_ptr, 0, (B.m + 1) * sizeof(int));
  for (int k = 0; k < A.nnz; k++)
    B.col_ptr[A.col_idx[k] + 1]++;

  // Prefix sum to get column pointers
  for (int j = 1; j <= B.m; j++)
    B.col_ptr[j] += B.col_ptr[j - 1];

  // Scatter non-zeros into CSC order
  vector<int> offset(B.m, 0);
  for (int i = 0; i < A.n; i++) {
    for (int k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
      int j = A.col_idx[k];
      int dst = B.col_ptr[j] + offset[j];
      B.row_idx[dst] = i;
      B.val[dst] = A.val[k];
      offset[j]++;
    }
  }

  return B;
}

// CSC serial SpMV: y = A * x
// Iterate over columns, scatter val * x[j] to y[row_idx[k]]
void spmv_csc(const CSC &A, const float *x, float *y) {
  const int *cp = A.col_ptr;
  const int *ri = A.row_idx;
  const float *vl = A.val;

  memset(y, 0, A.n * sizeof(float));
  for (int j = 0; j < A.m; j++) {
    float xj = x[j];
    for (int k = cp[j]; k < cp[j + 1]; k++)
      y[ri[k]] += vl[k] * xj;
  }
}

// CSC OpenMP SpMV: y = A * x
// Hybrid strategy:
//   ≤8 threads: per-thread private buffers + reduction (low overhead, good scaling)
//   >8 threads: atomic addition (avoids O(N×nthreads) reduction collapse)
void spmv_csc_omp(const CSC &A, const float *x, float *y) {
  const int N = A.n, M = A.m;
  int nthreads = omp_get_max_threads();

  if (nthreads <= 8) {
    // -- Low-thread path: per-thread private buffers + reduction --
    // The reduction adds O(N×nthreads) work, acceptable for ≤8 threads.
    static float **bufs = nullptr;
    static int    *buf_caps = nullptr;
    static int     buf_nth = 0;

    memset(y, 0, N * sizeof(float));

#pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int nth = omp_get_num_threads();

#pragma omp single
      {
        if (nth != buf_nth) {
          if (bufs) {
            for (int t = 0; t < buf_nth; t++) free(bufs[t]);
            free(bufs);
            free(buf_caps);
          }
          bufs = (float **)aligned_alloc(64, nth * sizeof(float *));
          buf_caps = (int *)aligned_alloc(64, nth * sizeof(int));
          for (int t = 0; t < nth; t++) {
            bufs[t] = nullptr;
            buf_caps[t] = 0;
          }
          buf_nth = nth;
        }
      }

      if (buf_caps[tid] < N) {
        free(bufs[tid]);
        bufs[tid] = alloc_float(N);
        buf_caps[tid] = N;
      }

      float *y_priv = bufs[tid];
      memset(y_priv, 0, N * sizeof(float));

#pragma omp for schedule(static)
      for (int j = 0; j < M; j++) {
        float xj = x[j];
        for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; k++)
          y_priv[A.row_idx[k]] += A.val[k] * xj;
      }

#pragma omp for schedule(static)
      for (int i = 0; i < N; i++) {
        float sum = 0.0f;
        for (int t = 0; t < nth; t++)
          sum += bufs[t][i];
        y[i] = sum;
      }
    }
  } else {
    // -- High-thread path: atomic scatter --
    // Work is O(nnz), independent of thread count.
    memset(y, 0, N * sizeof(float));

#pragma omp parallel for schedule(static)
    for (int j = 0; j < M; j++) {
      float xj = x[j];
      for (int k = A.col_ptr[j]; k < A.col_ptr[j + 1]; k++) {
#pragma omp atomic
        y[A.row_idx[k]] += A.val[k] * xj;
      }
    }
  }
}

#endif
