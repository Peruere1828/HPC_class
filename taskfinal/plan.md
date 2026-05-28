# 大作业规划

## 背景分析

### 课程作业难度 (task1-7)

| 作业 | 内容 | 并行方法 |
|------|------|----------|
| task1 | MPI_Bcast 树形通信 | MPI Send/Recv |
| task2 | 环形收发 + 死锁解决 | MPI Send/Recv |
| task3 | 自定义数据类型发送对角线 | MPI_Type_indexed |
| task4 | MPI_Allgather 环形 vs Gather+Bcast | MPI Send/Recv |
| task5 | 蒙特卡洛求 Pi | MPI 树形规约 |
| task6 | 2D Poisson 方程 Gauss-Seidel 红黑排序 | MPI 域分解 + Sendrecv |
| task7 | 矩阵乘法循环顺序 (ijk/ikj/jki) + perf 分析 | OpenMP |

**整体难度**: MPI 基础通信 → 集体通信 → 域分解 → OpenMP 基础。最复杂的是 task6（2D 域分解 + 红黑排序）。

---

## 主题: 基于 CSR 格式的稀疏矩阵运算并行化 (SpMV)

### 为什么选这个

1. **自然延伸**: 从 task7 的稠密矩阵乘法出发，过渡到稀疏矩阵，同学们有直觉
2. **数学背景**: 有限元方法、图论、PageRank、机器学习都用到稀疏矩阵
3. **聚焦 CSR**: 不搞多种格式对比，把精力放在并行化策略和性能分析上
4. **难度适中**: 比 task7 稍难（稀疏数据结构 + 负载均衡 + MPI 通信），但不至于太出风头

### 矩阵生成

用 2D Laplacian 五点差分格式生成稀疏矩阵（N×N 网格 → N²×N² 矩阵，每行最多 5 个非零元）。这是有限元/有限差分的标准场景，数学学院同学能理解。

也可以支持随机稀疏矩阵（可控稀疏度）。

### 实现计划

#### 阶段 1: CSR 串行 SpMV
- CSR 数据结构: `values[]`, `col_idx[]`, `row_ptr[]`
- 生成 Laplacian 五点差分矩阵
- 串行 SpMV: `y = A * x`
- 与稠密矩阵乘法对比验证正确性

#### 阶段 2: OpenMP 并行 SpMV
- 按行分块: `#pragma omp parallel for` 直接并行
- 测试不同线程数 (1, 2, 4, 8)
- 不同矩阵规模 (N=500, 1000, 2000, 对应矩阵 250K, 1M, 4M 行)
- 分析: 稀疏矩阵是 memory-bound，加速比受限于内存带宽

#### 阶段 3: MPI 并行 SpMV
- 按行块划分矩阵到各进程（类似 task6 的域分解思路）
- 每个进程存储本地行块的 CSR
- 需要通信: 向量 x 中非本地行对应的元素
  - 用 MPI_Allgatherv 收集完整的 x，或
  - 只交换 ghost 元素（更高效，但实现复杂）
- 测试不同进程数的加速比

#### 阶段 4: 性能测试与分析
- SpMV: 不同规模 × 不同线程/进程数
- SpMM: 与 task7 稠密 GEMM 对比
- 用 perf 分析缓存行为（呼应 task7 的 L1-dcache-load-misses）
- 总结: 稀疏 vs 稠密的性能差异

### 讲解大纲 (~12 min)

1. **引言** (2 min): 什么是稀疏矩阵？哪里用到？（有限元、图、PageRank）
2. **CSR 格式** (2 min): 存储结构图示，与稠密矩阵对比
3. **SpMV 串行实现** (1 min): 算法 + 正确性验证
4. **OpenMP 并行 SpMV** (2 min): 按行分块、加速比曲线
5. **MPI 并行 SpMV** (2 min): 行划分 + 通信策略、加速比
6. **SpMM 扩展** (1 min): 稀疏 × 稠密，与 task7 对比
7. **性能分析** (1 min): perf 缓存分析、memory-bound 分析
8. **总结** (1 min): 稀疏矩阵并行化的要点

### 文件结构

```
taskfinal/
├── plan.md              # 本文件
├── README.md            # 作业说明 + 运行方法 + 结果
├── Makefile
├── csr.h                # CSR 数据结构 + 生成矩阵
├── spmv.cpp             # 串行 SpMV + OpenMP 并行 SpMV
└── spmv_mpi.cpp         # MPI 并行 SpMV
```

### 技术细节

#### CSR 格式
```
矩阵 A (4×4):
[ 1  0  2  0 ]      values[]  = [1, 2, 3, 4, 5, 6, 7]
[ 0  3  0  4 ]      col_idx[] = [0, 2, 1, 3, 0, 2, 3]
[ 5  0  6  0 ]      row_ptr[] = [0, 2, 4, 6, 7]
[ 0  0  0  7 ]
```

#### SpMV 核心
```c
// 串行
for (int i = 0; i < N; i++) {
    double sum = 0;
    for (int j = row_ptr[i]; j < row_ptr[i+1]; j++)
        sum += values[j] * x[col_idx[j]];
    y[i] = sum;
}

// OpenMP 并行
#pragma omp parallel for
for (int i = 0; i < N; i++) { ... }
```

#### MPI 并行策略
```
进程 0: 行 [0, N/p)       → 本地 CSR
进程 1: 行 [N/p, 2N/p)    → 本地 CSR
...
需要: 完整向量 x (MPI_Allgatherv 或点对点交换)
```

### 预期结果

- SpMV OpenMP 加速比: 受内存带宽限制，4 线程约 2-3x
- SpMV MPI 加速比: 通信开销使效率低于 OpenMP
- perf 分析: 稀疏矩阵 L1-dcache-load-misses 高于稠密 ikj 但低于稠密 ijk
