利用 `MPI_Send` 和 `MPI_Recv` 实现 `MPI_Bcast`。

采用树形通信实现。

运行：

```
make
mpirun -n 6 ./main
```

输出结果：

```
root=2 casting=12345
myrank=2 send to=3
myrank=2 send to=4
myrank=1 recv from=4
myrank=3 recv from=2
myrank=3 send to=5
myrank=3 send to=0
myrank=4 recv from=2
myrank=4 send to=1
myrank=5 recv from=3
myrank=0 recv from=3
myrank=0 receive=12345
myrank=5 receive=12345
myrank=2 receive=12345
myrank=1 receive=12345
myrank=3 receive=12345
myrank=4 receive=12345
```