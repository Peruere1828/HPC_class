#!/bin/bash
#BSUB -J spmv_bench
#BSUB -q 7702ib
#BSUB -n 20
#BSUB -R "span[hosts=1]"
#BSUB -M 4GB
#BSUB -W 02:00
#BSUB -o %J.out
#BSUB -e %J.err

# --- 1. Modules (no cmake needed, Eigen is bundled) ---
module purge
module load gcc/11.2.0
module load make/4.3

# --- 2. Build with Makefile ---
cd "$(dirname "$0")"
make clean
make -j${LSB_DJOB_NUMPROC:-4}

# --- 3. Matrix files ---
MTX_FILES="testcases/parabolic_fem/parabolic_fem.mtx \
           testcases/apache2/apache2.mtx \
           testcases/pre2/pre2.mtx \
           testcases/amazon0312/amazon0312.mtx \
           testcases/neos3/neos3.mtx \
           testcases/wheel_601/wheel_601.mtx"

# --- 4. Benchmark with varying thread counts ---
export OMP_PROC_BIND=true

for nth in 1 2 4 8 16 20; do
  export OMP_NUM_THREADS=$nth
  echo ""
  echo "================================================"
  echo "  Threads: $nth"
  echo "================================================"

  echo ""
  echo ">>> Built-in generators (spmv):"
  ./spmv $nth

  echo ""
  echo ">>> Built-in generators + Real matrices (spmv_eigen):"
  ./spmv_eigen $nth $MTX_FILES

  echo ""
  echo ">>> Built-in generators (spmv_csc):"
  ./spmv_csc $nth

  echo ""
  echo ">>> Built-in generators + Real matrices (spmv_csc_eigen):"
  ./spmv_csc_eigen $nth $MTX_FILES
done

echo ""
echo "Done."
