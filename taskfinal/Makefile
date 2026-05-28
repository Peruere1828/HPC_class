# 编译器设置
CXX     = g++
MPICXX  = mpicxx

# 编译选项
CXXFLAGS  = -O3 -fopenmp
MPICXXFLAGS = -O3 -fopenmp

TARGETS = spmv spmv_single spmv_mpi

all: $(TARGETS)

spmv: spmv.cpp csr.h
	$(CXX) $(CXXFLAGS) -o $@ $<

spmv_single: spmv_single.cpp csr.h
	$(CXX) $(CXXFLAGS) -o $@ $<

spmv_mpi: spmv_mpi.cpp csr.h
	$(MPICXX) $(MPICXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS) *.o

.PHONY: all clean
