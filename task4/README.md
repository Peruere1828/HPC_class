仅使用 `MPI_Send` 和 `MPI_Recv` 实现 `MPI_Allgather`，提供两种算法。

## 实现一：环形传递

将 P 个进程排成逻辑环，共进行 P-1 轮：

1. 每进程先将自身数据拷贝到接收缓冲区对应位置。
2. 每轮将上一轮收到的数据块发给右邻居，同时从左邻居接收新块，按发送者 rank 放入对应位置。
3. 偶数号进程先发后收，奇数号进程先收后发，避免死锁。

P-1 轮后每个进程都拥有全部进程的数据。每步各进程只收发一个块，通信负载均衡。

## 实现二：Gather + Bcast

将 Allgather 分解为两步：

1. Gather：非 root 进程将数据发送到 rank 0；rank 0 先复制自身数据，再顺序接收各进程数据拼成完整数组。
2. Bcast：按照上一次作业的实现，通过二叉树将完整数组广播到所有进程。

Gather 阶段 root 串行接收 P-1 个进程的数据，Bcast 阶段 root 需将 P 倍数据分发出去，root 为通信瓶颈。

## 运行

```
make
mpirun -n 6 ./main
```

## 输出结果

每进程贡献 4 个 int（`[myrank*100, myrank*100+1, ...]`），同时调用两种实现和内置 `MPI_Allgather` 对比，通过 `MPI_Reduce` 汇总各进程验证结果。

```
cpusize=6
from rank 0: 0 1 2 3 
from rank 1: 100 101 102 103 
from rank 2: 200 201 202 203 
from rank 3: 300 301 302 303 
from rank 4: 400 401 402 403 
from rank 5: 500 501 502 503 
ring allgather: PASS
gather+bcast allgather: PASS
```
