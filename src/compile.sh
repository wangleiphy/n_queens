#!/bin/sh

set -x

# Arch flags: 4090 sm_89 (default) | 5090 sm_120 | A100/A800 sm_80 | H100/H800/H200 sm_90 | V100 sm_70
# Override examples:
#   ARCH="-gencode arch=compute_120,code=sm_120" CONFIG=7 sh compile.sh
#   ARCH="-gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80" sh compile.sh   # fat binary
ARCH=${ARCH:-"-gencode arch=compute_89,code=sm_89"}
CONFIG=${CONFIG:-2}
nvcc $ARCH --ptxas-options=-v -allow-unsupported-compiler -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -lcudart n_queens_cuda.cu n_queens.cpp utils.cpp main.cpp -o n_queens -O3 -Xcompiler -fopenmp -D_USE_CONFIG${CONFIG}_
