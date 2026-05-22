#include <chrono>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace std;
using namespace chrono;

const int M = 1024;
const int N = 1024;
const int L = 1024;

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

int main(int argc, char **argv) {
  int reps = 20;
  if (argc > 1)
    reps = atoi(argv[1]);

  vector<double> A(M * N), B(N * L), C(M * L);
  for (int i = 0; i < M * N; i++)
    A[i] = rand() % 100 / 10.0;
  for (int i = 0; i < N * L; i++)
    B[i] = rand() % 100 / 10.0;

  using Clock = high_resolution_clock;

  for (int rep = 0; rep < reps; rep++) {
    fill(C.begin(), C.end(), 0.0);
    auto t0 = Clock::now();
    matmul_ikj(A, B, C);
    auto t1 = Clock::now();
    double dt = duration<double>(t1 - t0).count();
    printf("ikj  rep=%2d  time=%.4f s\n", rep, dt);
  }
  return 0;
}
