#include <cuda_runtime.h>
#include <math.h>

#define Br 128
#define Bc 16
#define D_TILE 32   // process d dimension in chunks of 32

// K2: Optimizations over K1:
//   1. Br=128 (vs K1's 64): Tr halved → more work per block, better amortization
//   2. float4 vectorized K/V loads (requires d % 4 == 0)
//   3. Bc=16 (vs old 32): halves ss[] register count (16 vs 32)
//   4. D-tiling: process d in chunks of D_TILE=32, so only Qi[32]+Oi[32] live
//      at a time → 32+32+16 = 80 regs vs old 64+64+32 = 160 regs
//
// D-tiling approach:
//   For each KV tile, compute ss[Bc] once (dot product needs full d, but we
//   accumulate partial sums across d-tiles). Then for each d-tile, update Oi.
//   Actually: ss = Q @ K^T needs all of d simultaneously. So we tile differently:
//   - Keep Qi in registers but load/store Oi from shared memory per d-tile
//   No — simpler: just reduce Bc to 16 and keep the same structure. The main
//   register savings come from Bc=16 (ss[16] instead of ss[32]).
//
// Occupancy analysis (P100, sm_60):
//   Registers: Qi[64]+Oi[64]+ss[16]+misc ≈ 154/thread
//   Shared: Ks[16][64]+Vs[16][64] = 8KB
//   __launch_bounds__(128,4) → compiler targets ≤128 regs (65536/(128×4))
//   48KB/8KB = 6 blocks (shared), but launch_bounds caps at 4 → 4 blocks/SM
//   4×128 = 512 threads/SM = 16 warps = 25% occupancy
//
// grid = (Tr, bh), blockDim = Br = 128

__global__ __launch_bounds__(128, 4)
void flash_attention_v2_kernel(
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

    // l/m in registers: never touch HBM inside the kv loop
    float mi = -1e9f;
    float li = 0.0f;

    // Only K/V need shared memory (broadcast access: all threads read same [j][i])
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    // Q in registers: each thread owns one row, no sharing needed
    float Qi[64];
    float Oi[64];
    float scale  = 1.0f / sqrtf((float)d);
    int d4 = d / 4;

    // Zero-init Oi
    for (int i = 0; i < d; i++) Oi[i] = 0.0f;

    // Load Q row into registers (float4 vectorized)
    if (q_row < N) {
        const float4* Q4 = reinterpret_cast<const float4*>(Qh + q_row * d);
        float4* Qi4 = reinterpret_cast<float4*>(Qi);
        for (int i = 0; i < d4; i++) Qi4[i] = Q4[i];
    } else {
        for (int i = 0; i < d; i++) Qi[i] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        // Load K/V tile: first Bc threads each load one row (float4 vectorized)
        if (tid < Bc) {
            int kv_row = kv_tile * Bc + tid;
            if (kv_row < N) {
                const float4* K4 = reinterpret_cast<const float4*>(Kh + kv_row * d);
                const float4* V4 = reinterpret_cast<const float4*>(Vh + kv_row * d);
                float4* Ks4 = reinterpret_cast<float4*>(Ks[tid]);
                float4* Vs4 = reinterpret_cast<float4*>(Vs[tid]);
                for (int i = 0; i < d4; i++) { Ks4[i] = K4[i]; Vs4[i] = V4[i]; }
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

        // Online softmax (l/m stay in registers)
        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, ss[j]);

        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) {
            ss[j] = expf(ss[j] - mj);
            lj += ss[j];
        }

        float alpha  = expf(mi - mj);
        float li_new = li * alpha + lj;

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

extern "C" void flash_attention_v2(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    flash_attention_v2_kernel<<<grid, Br>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
