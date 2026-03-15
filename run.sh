#!/bin/bash
# run.sh - build and run the FlashAttention benchmark
# Usage: ./run.sh [ARCH] [NVCC_PATH]
#   ARCH      : GPU architecture (default: sm_60 for P100)
#   NVCC_PATH : path to nvcc if not in PATH (default: auto-detect)

set -e

ARCH="${1:-sm_60}"
NVCC="${2:-$(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)}"

if [ ! -x "$NVCC" ]; then
    echo "Error: nvcc not found. Run 'module load cuda' or pass path as second argument."
    echo "  Usage: ./run.sh sm_60 /usr/local/cuda-10.2/bin/nvcc"
    exit 1
fi

echo "nvcc : $NVCC"
echo "arch : $ARCH"
echo ""

make bench NVCC="$NVCC" ARCH="$ARCH"
echo ""
./build/benchmark
