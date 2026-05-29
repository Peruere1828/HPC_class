# 编译器设置
CC      = gcc          # C 编译器
CXX     = g++         # C++ 编译器
FC      = gfortran        # Fortran90 编译器

# 编译选项（基础警告 + 优化）
CFLAGS  = -O3 -fopenmp
CXXFLAGS = -O3 -fopenmp
FFLAGS  = -O3 -fopenmp

# 最终生成的可执行文件名（可自己改）
TARGET  = LU

# 自动扫描当前目录所有源文件
SRC_C   = $(wildcard *.c)
SRC_CPP = $(wildcard *.cpp)
SRC_F90 = $(wildcard *.f90)

# 自动生成目标文件
OBJ_C   = $(SRC_C:.c=.o)
OBJ_CPP = $(SRC_CPP:.cpp=.o)
OBJ_F90 = $(SRC_F90:.f90=.o)
OBJS    = $(OBJ_C) $(OBJ_CPP) $(OBJ_F90)

# 默认目标
all: $(TARGET)

# 链接生成可执行文件
$(TARGET): $(OBJS)
	$(CXX) -fopenmp -o $@ $^

# 编译 C 文件
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 编译 C++ 文件
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# 编译 Fortran90 文件
%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

json: clean
	bear -- make
	mv compile_commands.json ../.vscode/compile_commands.json

clean:
	rm -f $(OBJS) $(TARGET) compile_commands.json *.mod

.PHONY: all clean json