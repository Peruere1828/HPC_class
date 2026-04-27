#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

int cpusize, myrank;

void my_MPI_Bcast(void *data, int count, MPI_Datatype datatype, int root) {
  int rel_rank = (myrank - root + cpusize) % cpusize;
  if (rel_rank != 0) {
    int parent = ((rel_rank - 1) / 2 + root) % cpusize;
    MPI_Recv(data, count, datatype, parent, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    printf("myrank=%d recv from=%d\n", myrank, parent);
  }

  int rel_child1 = 2 * rel_rank + 1;
  int rel_child2 = 2 * rel_rank + 2;

  if (rel_child1 < cpusize) {
    int child1 = (rel_child1 + root) % cpusize;
    MPI_Send(data, count, datatype, child1, 0, MPI_COMM_WORLD);
    printf("myrank=%d send to=%d\n", myrank, child1);
  }
  if (rel_child2 < cpusize) {
    int child2 = (rel_child2 + root) % cpusize;
    MPI_Send(data, count, datatype, child2, 0, MPI_COMM_WORLD);
    printf("myrank=%d send to=%d\n", myrank, child2);
  }
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int value = 0;
  const int root = 2;

  if (myrank == root) {
    value = 12345;
    printf("root=%d casting=%d\n", myrank, value);
  }

  my_MPI_Bcast(&value, 1, MPI_INT, root);

  MPI_Barrier(MPI_COMM_WORLD);

  printf("myrank=%d receive=%d\n", myrank, value);
  MPI_Finalize();
  return 0;
}
