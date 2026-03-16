#include <cuda_runtime.h>
#include <math.h>

#define Br 128
#define Bc 32

// K2: Optimizations over K1:
//   1. Br=128 (vs K1's 64): Tr halved → more work per block, better amortization
//   2. float4 vectorized K/V loads (requires d % 4 == 0)
//
// Occupancy analysis (P100, sm_60):
//   Registers: Qi[64]+Oi[64]+ss[32]+misc ≈ 170/thread
//   Shared: Ks[32][64]+Vs[32][64] = 16KB
//   __launch_bounds__(128,3) → compiler targets ≤170 regs (65536/(128×3) = 170)
//   48KB/16KB = 3 blocks (shared) → 3 blocks/SM
//   3×128 = 384 threads/SM = 12 warps = 18.75% occupancy
//
// Without __launch_bounds__, compiler may use ~180-200 regs → only 2 blocks → 12.5%
// The hint saves ~33% occupancy and prevents the massive K2 slowdown at large N.
//
// grid = (Tr, bh), blockDim = Br = 128

__global__ __launch_bounds__(128, 3)
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
    float Oi[64] = {0.0f};
    float scale  = 1.0f / sqrtf((float)d);
    int d4 = d / 4;

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
