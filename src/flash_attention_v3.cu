#include <cuda_runtime.h>
#include <math.h>

#define Br 128
#define Bc 16

// K3: Optimization over K2 — cooperative K/V loading across all threads.
//
// In K2, only the first Bc=16 threads load K/V (the other 112 idle during loading).
// Here, all Br=128 threads participate via linear-index cooperative loading:
//   for (idx = tid; idx < Bc*d; idx += Br)
//     Ks[idx/d][idx%d] = Kh[...]
//
// Benefits:
//   1. 8x faster loading (128 threads vs 16)
//   2. Naturally coalesced: consecutive threads access consecutive addresses
//   3. Naturally bank-conflict-free for writes: consecutive threads write
//      consecutive shared memory addresses → each thread hits a different bank
//      (bank = idx % 32, and consecutive tids have consecutive idx values)
//
// Also uses Bc=16 (matching K2) for reduced register pressure (ss[16]).
//
// Occupancy analysis (P100, sm_60):
//   Registers: Qi[64]+Oi[64]+ss[16]+misc ≈ 154/thread
//   Shared: Ks[16][64]+Vs[16][64] = 8KB
//   __launch_bounds__(128,4) → compiler targets ≤128 regs
//   48KB/8KB = 6 blocks (shared), launch_bounds caps at 4 → 4 blocks/SM
//   4×128 = 512 threads/SM = 16 warps = 25% occupancy
//
// grid = (Tr, bh), blockDim = Br = 128

__global__ __launch_bounds__(128, 4)
void flash_attention_v3_kernel(
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

    int tid   = threadIdx.x;
    int q_row = q_tile * Br + tid;

    float mi = -1e9f;
    float li = 0.0f;

    // Shared memory for K/V tiles — only 8KB total with Bc=16
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    float Qi[64];
    float Oi[64];
    float scale  = 1.0f / sqrtf((float)d);
    int d4 = d / 4;

    // Zero-init Oi
    for (int i = 0; i < d; i++) Oi[i] = 0.0f;

    // Load Q row into registers (float4)
    if (q_row < N) {
        const float4* Q4 = reinterpret_cast<const float4*>(Qh + q_row * d);
        float4* Qi4 = reinterpret_cast<float4*>(Qi);
        for (int i = 0; i < d4; i++) Qi4[i] = Q4[i];
    } else {
        for (int i = 0; i < d; i++) Qi[i] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        // Cooperative load: all Br=128 threads share the work of loading
        // Bc*d = 16*64 = 1024 values → 8 values per thread
        // Linear indexing: thread tid loads elements tid, tid+128, tid+256, ...
        // → consecutive threads access consecutive addresses → coalesced + no bank conflict
        int tile_base = kv_tile * Bc;
        for (int idx = tid; idx < Bc * d; idx += Br) {
            int r = idx / d;
            int c = idx % d;
            int kv_row = tile_base + r;
            float kval = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
            float vval = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
            Ks[r][c] = kval;
            Vs[r][c] = vval;
        }
        __syncthreads();

        // S = Q @ K^T * scale in registers
        float ss[Bc];
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += Qi[i] * Ks[j][i];
            ss[j] = s * scale;
        }

        // Online softmax
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

extern "C" void flash_attention_v3(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    flash_attention_v3_kernel<<<grid, Br>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
