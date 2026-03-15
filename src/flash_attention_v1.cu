#include <cuda_runtime.h>
#include <math.h>

#define Br 64
#define Bc 32

// K1: Basic FlashAttention — Tiling + Online Softmax
//
// Key design vs flash-attention-minimal (Br=32):
//   - Br=64 (> d=64) so that Tr = N/64, halving K/V re-reads from HBM
//     (at Br=32, d=64, total DRAM ≈ 4N² = same as naive → no benefit)
//   - Ss in registers: each thread only reads its own scores, no sharing needed
//
// What's NOT optimized here (saved for K2):
//   - Qs still in shared memory (could be registers since each thread owns one row)
//   - l/m still round-trip through HBM each kv_tile
//   - No float4 vectorized loads
//
// grid = (Tr, bh): blockIdx.x = Q tile, blockIdx.y = head
// blockDim = Br = 64 threads; first Bc=32 threads load K/V tiles

__global__ void flash_attention_v1_kernel(
    const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
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
    float* lh = l + bh_idx * N;
    float* mh = m + bh_idx * N;

    int tid   = threadIdx.x;   // 0..Br-1
    int q_row = q_tile * Br + tid;

    // Qs in shared memory: basic approach (K2 moves it to registers)
    // Ss NOT in shared memory: each thread only reads its own row → registers
    __shared__ float Qs[Br][64];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    float Oi[64] = {0.0f};
    float scale  = 1.0f / sqrtf((float)d);

    // Load Q tile: each of Br threads loads its own row
    if (q_row < N) {
        for (int i = 0; i < d; i++) Qs[tid][i] = Qh[q_row * d + i];
    } else {
        for (int i = 0; i < d; i++) Qs[tid][i] = 0.0f;
    }

    // K1: l/m in HBM, initialized before the kv loop
    if (q_row < N) { lh[q_row] = 0.0f; mh[q_row] = -1e9f; }

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

        // S = Q @ K^T * scale — scores stay in registers (thread-private)
        float ss[Bc];
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += Qs[tid][i] * Ks[j][i];
            ss[j] = s * scale;
        }

        // Online softmax: read l/m from HBM each kv_tile
        float mi = (q_row < N) ? mh[q_row] : -1e9f;
        float li = (q_row < N) ? lh[q_row] : 0.0f;

        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, ss[j]);

        // Reuse ss[] for exp values
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

        // Write l/m back to HBM
        if (q_row < N) { mh[q_row] = mj; lh[q_row] = li_new; }
        __syncthreads();
    }

    if (q_row < N) {
        float li_final = lh[q_row];
        for (int i = 0; i < d; i++) Oh[q_row * d + i] = Oi[i] / li_final;
    }
}

extern "C" void flash_attention_v1(
    const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    flash_attention_v1_kernel<<<grid, Br>>>(Q, K, V, O, l, m, bh, N, d);
    cudaDeviceSynchronize();
}
