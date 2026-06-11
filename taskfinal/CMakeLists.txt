cmake_minimum_required(VERSION 3.14)
project(SpMV_Bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# --- OpenMP ---
find_package(OpenMP REQUIRED)

# --- Eigen (auto download via FetchContent) ---
include(FetchContent)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(eigen)

# --- Executables ---
add_executable(spmv spmv.cpp csr.h mmio.h)
target_link_libraries(spmv PRIVATE OpenMP::OpenMP_CXX)

add_executable(spmv_eigen spmv_eigen.cpp csr.h)
target_link_libraries(spmv_eigen PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)

add_executable(spmv_csc spmv_csc.cpp csr.h csc.h mmio.h)
target_link_libraries(spmv_csc PRIVATE OpenMP::OpenMP_CXX)

add_executable(spmv_csc_eigen spmv_csc_eigen.cpp csr.h csc.h)
target_link_libraries(spmv_csc_eigen PRIVATE OpenMP::OpenMP_CXX Eigen3::Eigen)
