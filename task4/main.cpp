#include <bits/stdc++.h>
#include <mpi.h>
using namespace std;

int cpusize, myrank;

// ring allgather
void my_MPI_Allgather_ring(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype) {
  int type_size;
  MPI_Type_size(sendtype, &type_size);

  memcpy((char *)recvbuf + myrank * recvcount * type_size, sendbuf,
         sendcount * type_size);

  int prev = (myrank + cpusize - 1) % cpusize;
  int next = (myrank + 1) % cpusize;

  for (int step = 1; step < cpusize; step++) {
    int send_src = (myrank - step + 1 + cpusize) % cpusize;
    int recv_src = (myrank - step + cpusize) % cpusize;

    if (myrank % 2 == 0) {
      MPI_Send((char *)recvbuf + send_src * recvcount * type_size, sendcount,
               sendtype, next, 0, MPI_COMM_WORLD);
      MPI_Recv((char *)recvbuf + recv_src * recvcount * type_size, recvcount,
               recvtype, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv((char *)recvbuf + recv_src * recvcount * type_size, recvcount,
               recvtype, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send((char *)recvbuf + send_src * recvcount * type_size, sendcount,
               sendtype, next, 0, MPI_COMM_WORLD);
    }
  }
}

// gather + bcast
void my_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int root) {
  int type_size;
  MPI_Type_size(sendtype, &type_size);

  if (myrank == root) {
    memcpy((char *)recvbuf + root * recvcount * type_size, sendbuf,
           sendcount * type_size);
    for (int i = 0; i < cpusize; i++) {
      if (i == root)
        continue;
      MPI_Recv((char *)recvbuf + i * recvcount * type_size, recvcount, recvtype,
               i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  } else {
    MPI_Send(sendbuf, sendcount, sendtype, root, 0, MPI_COMM_WORLD);
  }
}

void my_MPI_Bcast(void *data, int count, MPI_Datatype datatype, int root) {
  int rel_rank = (myrank - root + cpusize) % cpusize;
  if (rel_rank != 0) {
    int parent = ((rel_rank - 1) / 2 + root) % cpusize;
    MPI_Recv(data, count, datatype, parent, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }

  int rel_child1 = 2 * rel_rank + 1;
  int rel_child2 = 2 * rel_rank + 2;

  if (rel_child1 < cpusize) {
    int child1 = (rel_child1 + root) % cpusize;
    MPI_Send(data, count, datatype, child1, 0, MPI_COMM_WORLD);
  }
  if (rel_child2 < cpusize) {
    int child2 = (rel_child2 + root) % cpusize;
    MPI_Send(data, count, datatype, child2, 0, MPI_COMM_WORLD);
  }
}

void my_MPI_Allgather_gb(const void *sendbuf, int sendcount,
                         MPI_Datatype sendtype, void *recvbuf, int recvcount,
                         MPI_Datatype recvtype) {
  my_MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, 0);
  my_MPI_Bcast(recvbuf, cpusize * recvcount, recvtype, 0);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  const int N = 4;
  vector<int> sendbuf(N);
  vector<int> recv_ring(cpusize * N), recv_gb(cpusize * N),
      recv_ref(cpusize * N);

  for (int i = 0; i < N; i++)
    sendbuf[i] = myrank * 100 + i;

  my_MPI_Allgather_ring(sendbuf.data(), N, MPI_INT, recv_ring.data(), N,
                        MPI_INT);
  my_MPI_Allgather_gb(sendbuf.data(), N, MPI_INT, recv_gb.data(), N, MPI_INT);
  MPI_Allgather(sendbuf.data(), N, MPI_INT, recv_ref.data(), N, MPI_INT,
                MPI_COMM_WORLD);

  bool ring_ok = (recv_ring == recv_ref);
  bool gb_ok = (recv_gb == recv_ref);

  if (myrank == 0) {
    printf("cpusize=%d\n", cpusize);
    for (int r = 0; r < cpusize; r++) {
      printf("from rank %d: ", r);
      for (int i = 0; i < N; i++)
        printf("%d ", recv_ring[r * N + i]);
      printf("\n");
    }
  }

  int all_ok = 0;
  MPI_Reduce(&ring_ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
  if (myrank == 0)
    printf("ring allgather: %s\n", all_ok ? "PASS" : "FAIL");

  all_ok = 0;
  MPI_Reduce(&gb_ok, &all_ok, 1, MPI_INT, MPI_LAND, 0, MPI_COMM_WORLD);
  if (myrank == 0)
    printf("gather+bcast allgather: %s\n", all_ok ? "PASS" : "FAIL");

  MPI_Finalize();
  return 0;
}
