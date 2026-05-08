#include <ctype.h>
#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char byte;

int cpusize, myrank;

// ring allgather
void my_MPI_Allgather_ring(const void *sendbuf, int sendcount,
                           MPI_Datatype sendtype, void *recvbuf, int recvcount,
                           MPI_Datatype recvtype) {
  int type_size;
  MPI_Type_size(sendtype, &type_size);

  memcpy((char *)recvbuf + (size_t)myrank * recvcount * type_size, sendbuf,
         (size_t)sendcount * type_size);

  int prev = (myrank + cpusize - 1) % cpusize;
  int next = (myrank + 1) % cpusize;

  for (int step = 1; step < cpusize; step++) {
    int send_src = (myrank - step + 1 + cpusize) % cpusize;
    int recv_src = (myrank - step + cpusize) % cpusize;

    if (myrank % 2 == 0) {
      MPI_Send((char *)recvbuf + (size_t)send_src * recvcount * type_size, sendcount,
               sendtype, next, 0, MPI_COMM_WORLD);
      MPI_Recv((char *)recvbuf + (size_t)recv_src * recvcount * type_size, recvcount,
               recvtype, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else {
      MPI_Recv((char *)recvbuf + (size_t)recv_src * recvcount * type_size, recvcount,
               recvtype, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Send((char *)recvbuf + (size_t)send_src * recvcount * type_size, sendcount,
               sendtype, next, 0, MPI_COMM_WORLD);
    }
  }
}

// gather + bcast (binary tree)
void my_MPI_Gather(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                   void *recvbuf, int recvcount, MPI_Datatype recvtype,
                   int root) {
  int type_size;
  MPI_Type_size(sendtype, &type_size);

  if (myrank == root) {
    memcpy((char *)recvbuf + (size_t)root * recvcount * type_size, sendbuf,
           sendcount * type_size);
    for (int i = 0; i < cpusize; i++) {
      if (i == root)
        continue;
      MPI_Recv((char *)recvbuf + (size_t)i * recvcount * type_size, recvcount, recvtype,
               i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  } else {
    MPI_Send(sendbuf, sendcount, sendtype, root, 0, MPI_COMM_WORLD);
  }
}

void my_MPI_Bcast(void *data, size_t count, MPI_Datatype datatype, int root) {
  int rel_rank = (myrank - root + cpusize) % cpusize;

  size_t offset = 0;
  while (offset < count) {
    int chunk = (count - offset > (size_t)INT_MAX) ? INT_MAX : (int)(count - offset);

    if (rel_rank != 0) {
      int parent = ((rel_rank - 1) / 2 + root) % cpusize;
      MPI_Recv((char *)data + offset, chunk, datatype, parent, 0, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
    }

    int rel_child1 = 2 * rel_rank + 1;
    int rel_child2 = 2 * rel_rank + 2;

    if (rel_child1 < cpusize) {
      int child1 = (rel_child1 + root) % cpusize;
      MPI_Send((char *)data + offset, chunk, datatype, child1, 0, MPI_COMM_WORLD);
    }
    if (rel_child2 < cpusize) {
      int child2 = (rel_child2 + root) % cpusize;
      MPI_Send((char *)data + offset, chunk, datatype, child2, 0, MPI_COMM_WORLD);
    }

    offset += chunk;
  }
}

void my_MPI_Allgather_gb(const void *sendbuf, int sendcount,
                         MPI_Datatype sendtype, void *recvbuf, int recvcount,
                         MPI_Datatype recvtype) {
  my_MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, 0);
  my_MPI_Bcast(recvbuf, (size_t)cpusize * recvcount, recvtype, 0);
}

/* 校验: 第 j 块的值应等于 j + 1 */
static void check(int nprocs, int myrank, size_t size, byte *buffer) {
  size_t i, j;

  for (j = 0; j < nprocs; j++)
    for (i = 0; i < size; i++)
      if (buffer[j * size + i] != ((j + 1) & 255)) {
        fprintf(stderr,
                "Process %d: incorrect value at block %zu, "
                "position %zu\n",
                myrank, j, i);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
}

int main(int argc, char **argv) {
  int nprocs;
  byte *send_buffer, *recv_buffer;
  size_t size = 0;
  double time0, time1;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  cpusize = nprocs;

  if (argc != 2) {
    if (myrank == 0)
      fprintf(stderr, "Usage:   %s buffersize[K|M|G]\n", argv[0]);
    MPI_Finalize();
    exit(1);
  } else {
    char *p;
    size = strtol(argv[1], &p, 10);
    switch (toupper(*p)) {
    case 'G':
      size *= 1024;
    case 'M':
      size *= 1024;
    case 'K':
      size *= 1024;
      break;
    }
  }
  if (size <= 0) {
    fprintf(stderr, "Process %d: invalid size %zu\n", myrank, size);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  if (myrank == 0) {
    fprintf(stderr, "Allgather with %d processes, buffer size: %zu bytes\n",
            nprocs, size);
  }

  send_buffer = (byte *)malloc(size);
  recv_buffer = (byte *)malloc((size_t)nprocs * size);
  if (send_buffer == NULL || recv_buffer == NULL) {
    fprintf(stderr, "Process %d: memory allocation error!\n", myrank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  memset(send_buffer, myrank + 1, size);

  // Ring algorithm
  memset(recv_buffer, 0, (size_t)nprocs * size);
  MPI_Barrier(MPI_COMM_WORLD);
  time0 = MPI_Wtime();
  my_MPI_Allgather_ring(send_buffer, size, MPI_BYTE, recv_buffer, size,
                        MPI_BYTE);
  MPI_Barrier(MPI_COMM_WORLD);
  time1 = MPI_Wtime();
  if (myrank == 0)
    fprintf(stderr, "The circular algorithm: wall time = %lf\n", time1 - time0);
  check(nprocs, myrank, size, recv_buffer);

  // Gather+Bcast algorithm
  memset(recv_buffer, 0, (size_t)nprocs * size);
  MPI_Barrier(MPI_COMM_WORLD);
  time0 = MPI_Wtime();
  my_MPI_Allgather_gb(send_buffer, size, MPI_BYTE, recv_buffer, size, MPI_BYTE);
  MPI_Barrier(MPI_COMM_WORLD);
  time1 = MPI_Wtime();
  if (myrank == 0)
    fprintf(stderr, "Gather+Bcast: wall time = %lf\n", time1 - time0);
  check(nprocs, myrank, size, recv_buffer);

  // MPI_Allgather
  memset(recv_buffer, 0, (size_t)nprocs * size);
  MPI_Barrier(MPI_COMM_WORLD);
  time0 = MPI_Wtime();
  MPI_Allgather(send_buffer, size, MPI_BYTE, recv_buffer, size, MPI_BYTE,
                MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);
  time1 = MPI_Wtime();
  if (myrank == 0)
    fprintf(stderr, "MPI_Allgather: wall time = %lf\n", time1 - time0);
  check(nprocs, myrank, size, recv_buffer);

  free(send_buffer);
  free(recv_buffer);

  MPI_Finalize();
  return 0;
}
