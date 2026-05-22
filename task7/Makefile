# 编译器设置
CC      = mpicc          # C 编译器
CXX     = mpicxx         # C++ 编译器
FC      = mpifort        # Fortran90 编译器

# 编译选项（基础警告 + 优化）
CFLAGS  = -O3
CXXFLAGS = -O3 -fopenmp
FFLAGS  = -O3

CXXFLAGS = -O3 -fopenmp

TARGETS = main sub_ijk sub_ikj sub_jki

all: $(TARGETS)

main: main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

sub_ijk: sub_ijk.cpp
	$(CXX) -O3 -o $@ $<

sub_ikj: sub_ikj.cpp
	$(CXX) -O3 -o $@ $<

sub_jki: sub_jki.cpp
	$(CXX) -O3 -o $@ $<

json: clean
	bear -- make
	mv compile_commands.json ../.vscode/compile_commands.json

clean:
	rm -f $(TARGETS) *.o compile_commands.json *.mod

.PHONY: all clean json