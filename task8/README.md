LU 分解 OpenMP 并行。

原来的写法是 i-k-j 循环，这会导致数据竞争，因为行 $i$ 依赖所有 $k<i$ 的行已完成分解。改为 k-i-j 顺序，外层 k 串行，内层 i 和 j 循环用 OpenMP 并行。每个 `k` 步内线程只读已固定的第 `k` 行，消除跨行依赖。

运行：

```
make
export OMP_NUM_THREADS=20
./LU
```

输出：

```
N = 1024, threads = 20
elapsed time 6.80e-02, err= 3.33e-16
```