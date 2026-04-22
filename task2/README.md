利用 `MPI_Send` 和 `MPI_Recv` 实现计算 $\pi$。

我采取 `mt19937` 和 `uniform_real_distribution` 生成随机坐标，并使用树形规约加速通信。

运行：

```
make
mpirun -n 6 ./main
```