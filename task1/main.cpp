#include <bits/stdc++.h>
#include <mpi/mpi.h>
int cpusize, myrank;

void ring_send_recv(int buf_size) {
  int prev = (myrank + cpusize - 1) % cpusize, next = (myrank + 1) % cpusize;
  std::vector<int> send_buf(buf_size, myrank), recv_buf(buf_size);
  if (myrank % 2 == 0) {
    MPI_Send(send_buf.data(), buf_size, MPI_INT, next, 0, MPI_COMM_WORLD);
    MPI_Recv(recv_buf.data(), buf_size, MPI_INT, prev, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(recv_buf.data(), buf_size, MPI_INT, prev, 0, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Send(send_buf.data(), buf_size, MPI_INT, next, 0, MPI_COMM_WORLD);
  }

  std::cout << "进程 " << myrank << " | 发送到 " << next << " | 收到来自 "
            << prev << " 的数据首项: " << recv_buf[0] << std::endl;
}

void dead_ring_send_recv(int buf_size) {
  // 所有进程一起发送会导致死锁
  int prev = (myrank + cpusize - 1) % cpusize, next = (myrank + 1) % cpusize;
  std::vector<int> send_buf(buf_size, myrank), recv_buf(buf_size);

  MPI_Send(send_buf.data(), buf_size, MPI_INT, next, 0, MPI_COMM_WORLD);
  MPI_Recv(recv_buf.data(), buf_size, MPI_INT, prev, 0, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);

  std::cout << "进程 " << myrank << " | 发送到 " << next << " | 收到来自 "
            << prev << " 的数据首项: " << recv_buf[0] << std::endl;
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &cpusize);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  const int BUF_LEN = 10000000;

  ring_send_recv(BUF_LEN);

  // dead_ring_send_recv(BUF_LEN);

  MPI_Finalize();
  return 0;
}