#include <cuda_runtime.h>
#include <math.h>

#define Br 64
#define Bc 16

// K1: Basic FlashAttention — Tiling + Online Softmax
//
// Optimizations (vs naive):
//   - Tiling: Q in tiles of Br, K/V in tiles of Bc → O(N*d) DRAM, no N×N matrix
//   - Online softmax: running max/sum in registers, single pass over K/V tiles
//   - Qi in registers: each thread owns one Q row → no shared memory for Q
//   - l/m in registers: no HBM round-trips inside the kv loop
//   - Bc=16: reduces ss[] from 32 to 16 registers, allowing more blocks/SM
//
// What's NOT optimized here (saved for K2+):
//   - No float4 vectorized loads
//   - K/V loading: only first Bc threads load (no cooperative loading)
//
// Occupancy analysis (P100, sm_60):
//   Registers: Qi[64]+Oi[64]+ss[16]+misc ≈ 154/thread
//   Shared: Ks[16][64]+Vs[16][64] = 8KB
//   __launch_bounds__(64,6) → compiler targets ≤170 regs
//   65536/(154×64) = 6.6 → 6 blocks (regs), 48KB/8KB = 6 blocks (shared) → 6 blocks/SM
//   6×64 = 384 threads/SM = 12 warps = 18.75% occupancy
//
// grid = (Tr, bh): blockIdx.x = Q tile, blockIdx.y = head
// blockDim = Br = 64 threads; first Bc=16 threads load K/V tiles

__global__ __launch_bounds__(64, 6)
void flash_attention_v1_kernel(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int bh_idx = blockIdx.y;
    int q_tile = blockIdx.x;
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;
    if (bh_idx >= bh || q_tile >= Tr) return;

    const float* Qh = Q + (size_t)bh_idx * N * d;
    const float* Kh = K + (size_t)bh_idx * N * d;
    const float* Vh = V + (size_t)bh_idx * N * d;
    float* Oh = O + (size_t)bh_idx * N * d;

    int tid   = threadIdx.x;   // 0..Br-1
    int q_row = q_tile * Br + tid;

    // l/m in registers: no HBM round-trips
    float mi = -1e9f;
    float li = 0.0f;

    // Only K/V in shared memory (broadcast reads: all threads read same [j][i])
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    // Q in registers: each thread owns one row, no sharing needed
    float Qi[64];
    float Oi[64];
    float scale  = 1.0f / sqrtf((float)d);

    // Zero-init Oi
    for (int i = 0; i < d; i++) Oi[i] = 0.0f;

    // Load Q row into registers
    if (q_row < N) {
        for (int i = 0; i < d; i++) Qi[i] = Qh[q_row * d + i];
    } else {
        for (int i = 0; i < d; i++) Qi[i] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        // Load K/V tile: first Bc threads each load one row
        if (tid < Bc) {
            int kv_row = kv_tile * Bc + tid;
            if (kv_row < N) {
                for (int i = 0; i < d; i++) {
                    Ks[tid][i] = Kh[kv_row * d + i];
                    Vs[tid][i] = Vh[kv_row * d + i];
                }
            } else {
                for (int i = 0; i < d; i++) { Ks[tid][i] = 0.0f; Vs[tid][i] = 0.0f; }
            }
        }
        __syncthreads();

        // S = Q @ K^T * scale — scores in registers (thread-private)
        float ss[Bc];
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += Qi[i] * Ks[j][i];
            ss[j] = s * scale;
        }

        // Online softmax (l/m in registers — never touch HBM)
        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, ss[j]);

        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) {
            ss[j] = expf(ss[j] - mj);
            lj += ss[j];
        }

        float alpha  = expf(mi - mj);
        float li_new = li * alpha + lj;

        // Rescale accumulated O, then add this tile's contribution
        for (int i = 0; i < d; i++) Oi[i] *= alpha;
        for (int j = 0; j < Bc; j++) {
            float esj = ss[j];
            for (int i = 0; i < d; i++) Oi[i] += esj * Vs[j][i];
        }

        mi = mj;
        li = li_new;
        __syncthreads();
    }

    if (q_row < N) {
        for (int i = 0; i < d; i++) Oh[q_row * d + i] = Oi[i] / li;
    }
}

extern "C" void flash_attention_v1(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    flash_attention_v1_kernel<<<grid, Br>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
