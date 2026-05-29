#ifndef MMIO_H
#define MMIO_H

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include "csr.h"

// Read Matrix Market file into CSR format
// Supports: coordinate real/pattern (symmetric/unsymmetric)
CSR load_mm(const char *filename) {
  FILE *f = fopen(filename, "r");
  if (!f) { fprintf(stderr, "Cannot open %s\n", filename); exit(1); }

  char line[1024];
  bool symmetric = false;
  bool pattern = false;

  // skip comment lines, check header
  while (fgets(line, sizeof(line), f)) {
    if (line[0] != '%') break;
    if (strstr(line, "symmetric")) symmetric = true;
    if (strstr(line, "pattern")) pattern = true;
  }

  int M, N, nnz;
  sscanf(line, "%d %d %d", &M, &N, &nnz);

  // read COO triplets
  struct Trip { int r, c; float v; };
  std::vector<Trip> trips;
  trips.reserve(symmetric ? nnz * 2 : nnz);

  for (int i = 0; i < nnz; i++) {
    int r, c;
    if (pattern) {
      fscanf(f, "%d %d", &r, &c);
      r--; c--;
      trips.push_back({r, c, 1.0f});
      if (symmetric && r != c)
        trips.push_back({c, r, 1.0f});
    } else {
      double v;
      fscanf(f, "%d %d %lf", &r, &c, &v);
      r--; c--;
      trips.push_back({r, c, (float)v});
      if (symmetric && r != c)
        trips.push_back({c, r, (float)v});
    }
  }
  fclose(f);

  // sort by row then col
  std::sort(trips.begin(), trips.end(), [](const Trip &a, const Trip &b) {
    return a.r < b.r || (a.r == b.r && a.c < b.c);
  });

  // build CSR
  CSR A;
  A.n = M;
  A.m = N;
  A.nnz = (int)trips.size();
  A.row_ptr = alloc_int(M + 1);
  A.col_idx = alloc_int(A.nnz);
  A.val = alloc_float(A.nnz);

  memset(A.row_ptr, 0, (M + 1) * sizeof(int));
  for (int i = 0; i < A.nnz; i++)
    A.row_ptr[trips[i].r + 1]++;

  // prefix sum
  for (int i = 0; i < M; i++)
    A.row_ptr[i + 1] += A.row_ptr[i];

  for (int i = 0; i < A.nnz; i++) {
    A.col_idx[i] = trips[i].c;
    A.val[i] = trips[i].v;
  }

  return A;
}

#endif
