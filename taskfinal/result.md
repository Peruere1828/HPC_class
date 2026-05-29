# SpMV Benchmark Results

## 测试环境

- CPU: Intel/AMD x86_64 (4 threads)
- 编译器: GCC 13.3.0, -O3, OpenMP
- Eigen: 3.4.0 (FetchContent)
- 测试矩阵: SuiteSparse Matrix Collection

## 测试矩阵

| 矩阵          | 来源                                        | 大小 (N) | 非零元 (NNZ) | 类型             |
| ------------- | ------------------------------------------- | -------- | ------------ | ---------------- |
| parabolic_fem | Computational Fluid Dynamics Problem        | 525,825  | 3,674,625    | 对称             |
| apache2       | Structural Problem                          | 715,176  | 4,817,870    | 对称             |
| pre2          | Frequency Domain Circuit Simulation Problem | 659,033  | 5,959,282    | 非对称           |
| amazon0312    | Directed Graph                              | 400,727  | 3,200,440    | 非对称 (pattern) |
| neos3         | Linear Programming Problem                  | 512,209  | 2,055,024    | 非对称           |
| wheel_601     | Combinatorial Problem                       | 902,103  | 2,170,814    | 非对称           |

## 性能对比 (4线程)

| 矩阵          | CSR串行 (ms) | CSR+OpenMP (ms) | Eigen (ms) | OpenMP加速比 | Eigen加速比 |
| ------------- | ------------ | --------------- | ---------- | ------------ | ----------- |
| parabolic_fem | 1.86         | 0.82            | 1.21       | **2.3x**     | 1.5x        |
| apache2       | 2.44         | 1.22            | 1.51       | **2.0x**     | 1.6x        |
| pre2          | 3.79         | 1.49            | 1.78       | **2.5x**     | 2.1x        |
| amazon0312    | 3.33         | 0.95            | 1.28       | **3.5x**     | 2.6x        |
| neos3         | 1.20         | 0.37            | 0.72       | **3.2x**     | 1.7x        |
| wheel_601     | 2.36         | 0.74            | 1.24       | **3.2x**     | 1.9x        |

## 分析

### 1. OpenMP 加速比

- **最佳**: amazon0312 (3.5x) — 社交网络，行长度均匀，并行效率高
- **最差**: apache2 (2.0x) — 可能存在负载不均衡

### 2. Eigen vs 手写 CSR

- Eigen 使用 SIMD 优化和缓存友好的内存访问模式
- 对于结构化矩阵 (parabolic_fem, apache2)，Eigen 比串行 CSR 快 1.5-1.6x
- 对于稀疏矩阵 (pre2, amazon0312)，Eigen 优势更明显 (2.1-2.6x)

### 3. 结论

| 方法         | 优点                   | 缺点                 |
| ------------ | ---------------------- | -------------------- |
| CSR 串行     | 实现简单，内存占用最小 | 性能最低             |
| CSR + OpenMP | 加速比好，实现简单     | 受负载均衡影响       |
| Eigen        | SIMD优化，自动并行化   | 内存开销较大，编译慢 |

## 运行方法

```bash
cd taskfinal
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# 运行 benchmark
./spmv_eigen 4 ../testcases/parabolic_fem/parabolic_fem.mtx ../testcases/apache2/apache2.mtx ...
```
