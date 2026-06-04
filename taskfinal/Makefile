CXX      := g++
CXXFLAGS := -O3 -std=c++17 -march=native
OMPFLAGS := -fopenmp
EIGEN    := -I.

.PHONY: all clean

all: spmv spmv_eigen

spmv: spmv.cpp csr.h mmio.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ spmv.cpp

spmv_eigen: spmv_eigen.cpp csr.h mmio.h
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) $(EIGEN) -o $@ spmv_eigen.cpp

clean:
	rm -f spmv spmv_eigen