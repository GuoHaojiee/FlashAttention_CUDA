#include <cuda_runtime.h>
#include <math.h>

#define Br 128
#define Bc 32
#define WARP_D 2   // threads per Q row: split d dimension across 2 threads

// K4: Warp d-split tiling — reduce register pressure via d-dimension parallelism
//
// Problem in K3:
//   Each thread holds Qi[64] + Oi[64] + ss[32] = 160 registers (+ misc ≈ 170)
//   Even with __launch_bounds__(128,3), the compiler must aggressively manage 170 regs
//
// Fix: WARP_D=2 threads collaborate on each Q row, each holding half of d:
//   Thread 2k   holds Qi[0..31],  Oi[0..31]  → 32+32+32 = 96 regs
//   Thread 2k+1 holds Qi[32..63], Oi[32..63] → 32+32+32 = 96 regs
//   ss[Bc] is computed via __shfl_xor_sync to sum partial dot products
//
// Occupancy analysis (P100, sm_60):
//   Registers: ~110/thread
//   Shared: Ks[32][64]+Vs[32][64] = 16KB
//   __launch_bounds__(256,2) → 65536/(256×2) = 128 regs (plenty for ~110)
//   48KB/16KB = 3 blocks from shared, 2 blocks from launch_bounds → 2 blocks
//   2×256 = 512 threads/SM = 16 warps = 25% occupancy
//
// Layout:
//   blockDim = Br * WARP_D = 128 * 2 = 256 threads
//   q_idx = tid / WARP_D     → which Q row (0..127)
//   d_half = tid % WARP_D    → which half of d (0 or 1)
//
// grid = (Tr, bh), blockDim = 256

__global__ __launch_bounds__(256, 2)
void flash_attention_v4_kernel(
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

    int tid    = threadIdx.x;          // 0..255
    int q_idx  = tid / WARP_D;         // Q row within tile: 0..127
    int d_half = tid % WARP_D;         // 0 or 1
    int q_row  = q_tile * Br + q_idx;  // global Q row

    int half_d = d / WARP_D;           // 32
    int d_off  = d_half * half_d;      // offset into d dimension: 0 or 32

    float mi = -1e9f;
    float li = 0.0f;

    // K/V in shared memory (no padding: cooperative loading is bank-conflict-free)
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    // Each thread holds HALF of Q row and HALF of O row → 32+32 = 64 regs (vs K3's 128)
    float Qi_half[32];
    float Oi_half[32] = {0.0f};
    float scale = 1.0f / sqrtf((float)d);

    // Load Q half-row into registers
    if (q_row < N) {
        for (int i = 0; i < half_d; i++)
            Qi_half[i] = Qh[q_row * d + d_off + i];
    } else {
        for (int i = 0; i < half_d; i++)
            Qi_half[i] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        // Cooperative load: all 256 threads load Bc*d = 2048 values
        // → 8 values per thread (vs K3's 16 with 128 threads)
        int tile_base = kv_tile * Bc;
        int total_elems = Bc * d;
        int threads = Br * WARP_D;  // 256
        for (int idx = tid; idx < total_elems; idx += threads) {
            int r = idx / d;
            int c = idx % d;
            int kv_row = tile_base + r;
            float kval = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
            float vval = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
            Ks[r][c] = kval;
            Vs[r][c] = vval;
        }
        __syncthreads();

        // S = Q @ K^T * scale — each thread computes partial dot product over its half
        // Then reduce across WARP_D=2 partners via __shfl_xor_sync
        float ss[Bc];
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < half_d; i++)
                s += Qi_half[i] * Ks[j][d_off + i];
            // Sum partial dot products: thread 2k has sum of d[0..31],
            // thread 2k+1 has sum of d[32..63]. XOR with 1 swaps partners.
            s += __shfl_xor_sync(0xffffffff, s, 1);
            ss[j] = s * scale;
        }

        // Online softmax — both threads in a pair have identical ss[] values
        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, ss[j]);

        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) {
            ss[j] = expf(ss[j] - mj);
            lj += ss[j];
        }

        float alpha  = expf(mi - mj);
        float li_new = li * alpha + lj;

        // Update O: each thread updates its own half of d
        for (int i = 0; i < half_d; i++) Oi_half[i] *= alpha;
        for (int j = 0; j < Bc; j++) {
            float esj = ss[j];
            for (int i = 0; i < half_d; i++)
                Oi_half[i] += esj * Vs[j][d_off + i];
        }

        mi = mj;
        li = li_new;
        __syncthreads();
    }

    // Write output: each thread writes its half of the d dimension
    if (q_row < N) {
        for (int i = 0; i < half_d; i++)
            Oh[q_row * d + d_off + i] = Oi_half[i] / li;
    }
}

extern "C" void flash_attention_v4(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    int blockSize = Br * WARP_D;  // 256
    flash_attention_v4_kernel<<<grid, blockSize>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
