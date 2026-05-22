矩阵乘法循环顺序性能比较。

实现中把二维数组拍平成一维存放，方便函数传参。同时额外增加了一组使用 OpenMP 的 ikj 循环的测试。

运行：

```
make all
export OMP_NUM_THREADS=6
./main
```

输出：

```
GEMM loop-order comparison: A[1024][1024] x B[1024][1024] = C[1024][1024]
version          time(s)        MFLOPS       SPEEDUP     tr(C) check
ijk              69.2620        620.10          1.00              OK
ikj               3.2266      13311.02         21.47              OK
jki             189.5225        226.62          0.37              OK
ikj+omp           0.7539      56968.42         91.87              OK
```

perf结果：

运行：

```
make all
perf stat -d ./sub_ijk
perf stat -d ./sub_ikj
perf stat -d ./sub_jki
```

输出：

```
 Performance counter stats for './sub_ijk':

         66,175.43 msec task-clock                       #    1.000 CPUs utilized             
               312      context-switches                 #    4.715 /sec                      
                42      cpu-migrations                   #    0.635 /sec                      
             6,276      page-faults                      #   94.839 /sec                      
   305,446,002,016      cycles                           #    4.616 GHz                         (71.43%)
   237,580,010,956      stalled-cycles-frontend          #   77.78% frontend cycles idle        (71.43%)
   151,450,994,041      instructions                     #    0.50  insn per cycle            
                                                  #    1.57  stalled cycles per insn     (71.43%)
    21,697,210,025      branches                         #  327.874 M/sec                       (71.43%)
        42,724,227      branch-misses                    #    0.20% of all branches             (71.43%)
    43,819,846,205      L1-dcache-loads                  #  662.177 M/sec                       (71.43%)
    22,717,719,189      L1-dcache-load-misses            #   51.84% of all L1-dcache accesses   (71.43%)
   <not supported>      LLC-loads                                                             
   <not supported>      LLC-load-misses                                                       

      66.182065951 seconds time elapsed

      66.162741000 seconds user
       0.013998000 seconds sys

 Performance counter stats for './sub_ikj':

          5,042.42 msec task-clock                       #    1.000 CPUs utilized             
                28      context-switches                 #    5.553 /sec                      
                 9      cpu-migrations                   #    1.785 /sec                      
             6,277      page-faults                      #    1.245 K/sec                     
    23,230,658,163      cycles                           #    4.607 GHz                         (71.41%)
       831,029,687      stalled-cycles-frontend          #    3.58% frontend cycles idle        (71.41%)
    86,396,076,082      instructions                     #    3.72  insn per cycle            
                                                  #    0.01  stalled cycles per insn     (71.43%)
    10,835,324,387      branches                         #    2.149 G/sec                       (71.45%)
        23,563,800      branch-misses                    #    0.22% of all branches             (71.45%)
    32,443,362,705      L1-dcache-loads                  #    6.434 G/sec                       (71.44%)
     2,891,667,248      L1-dcache-load-misses            #    8.91% of all L1-dcache accesses   (71.42%)
   <not supported>      LLC-loads                                                             
   <not supported>      LLC-load-misses                                                       

       5.044464444 seconds time elapsed

       5.029580000 seconds user
       0.013998000 seconds sys

 Performance counter stats for './sub_jki':

        189,306.46 msec task-clock                       #    1.000 CPUs utilized             
               741      context-switches                 #    3.914 /sec                      
               167      cpu-migrations                   #    0.882 /sec                      
             6,276      page-faults                      #   33.153 /sec                      
   873,640,885,871      cycles                           #    4.615 GHz                         (71.43%)
   773,081,771,334      stalled-cycles-frontend          #   88.49% frontend cycles idle        (71.43%)
   174,336,461,661      instructions                     #    0.20  insn per cycle            
                                                  #    4.43  stalled cycles per insn     (71.43%)
    21,988,615,025      branches                         #  116.154 M/sec                       (71.43%)
        82,053,332      branch-misses                    #    0.37% of all branches             (71.43%)
    88,108,788,394      L1-dcache-loads                  #  465.429 M/sec                       (71.43%)
    96,505,625,636      L1-dcache-load-misses            #  109.53% of all L1-dcache accesses   (71.43%)
   <not supported>      LLC-loads                                                             
   <not supported>      LLC-load-misses                                                       

     189.322899129 seconds time elapsed

     189.280811000 seconds user
       0.029998000 seconds sys
```

观察发现 ijk 和 jki 的 `L1-dcache-load-misses` 异常高，说明缓存不命中率高。对比 ikj，可以发现缓存不命中率高直接导致 IPC 低，进而导致运行速度显著地慢于 ikj。

**结论**：

- 连续访存(ikj) vs 跨步长访存(ijk/jki)是最大影响因素，jki 比 ijk 更慢是因为 jki 还多一次跨步长访存
- ikj+omp 受内存带宽限制，多核加速比 ~4.27x 没有达到理论上限 6x
