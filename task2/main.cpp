#include <bits/stdc++.h>
#include <mpi.h>
#include <random>
using namespace std;

int cpusize, myrank;

const long long TOTAL_POINT = 6000000000;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  random_device rd;
  mt19937 gen(rd() + myrank * 9973);
  uniform_real_distribution<float> dist(0.0, 1.0);

  const long long POINT_COUNT = TOTAL_POINT / cpusize;
  long long in_circle = 0;
#pragma unroll
  for (long long i = 1; i <= POINT_COUNT; ++i) {
    float x = dist(gen), y = dist(gen);
    if (x * x + y * y <= 1)
      in_circle++;
  }

  long long total = in_circle;

  int step = 1;
  while (step < cpusize) {
    if (myrank % (2 * step) == 0) {
      if (myrank + step < cpusize) {
        long long buf;
        MPI_Recv(&buf, 1, MPI_LONG_LONG, myrank + step, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        total += buf;
      }
    } else {
      int target = myrank - step;
      MPI_Send(&total, 1, MPI_LONG_LONG, target, 0, MPI_COMM_WORLD);
      break;
    }
    step *= 2;
  }

  if (myrank == 0) {
    double ans = 4.0 * total / (POINT_COUNT * cpusize);
    cout << fixed << setprecision(10) << ans << endl;
  }
  MPI_Finalize();
  return 0;
}
