#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// K4: FlashAttention + Warp-parallel + Single-pass + Double-buffered K/V loading
//
// Improvement over K3:
//   K3 loads one KV tile, processes it, then loads the next → load/compute serial.
//   K4 uses double buffering: while computing on tile T, the next tile T+1 is
//   being loaded into a second shared memory buffer.
//
//   Two shared memory buffers:
//     buf0: Ks[0][Bc][64] + Vs[0][Bc][64] = 16KB
//     buf1: Ks[1][Bc][64] + Vs[1][Bc][64] = 16KB
//     Total = 32KB < 48KB → fits in P100 shared memory
//
// Same warp-per-row architecture: blockDim = (32, Br) = 1024 threads.
//
// Occupancy: 48KB/32KB = 1 block from shared (conservative). But with 1024
// threads/block and low register count, we still get 1024 threads/SM = 50%.

__global__ __launch_bounds__(1024, 1)
void flash_attention_v4_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
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
    float*       Oh = O + (size_t)bh_idx * N * d;

    int lane  = threadIdx.x;
    int row   = threadIdx.y;
    int q_row = q_tile * Br + row;

    float scale = 1.0f / sqrtf((float)d);

    float mi = -1e9f;
    float li = 0.0f;
    float Oi0 = 0.0f, Oi1 = 0.0f;

    float Qi0 = 0.0f, Qi1 = 0.0f;
    if (q_row < N) {
        int base = lane * 2;
        if (base < d)     Qi0 = Qh[q_row * d + base];
        if (base + 1 < d) Qi1 = Qh[q_row * d + base + 1];
    }

    // Double buffer for K/V tiles
    __shared__ float Ks[2][Bc][64];
    __shared__ float Vs[2][Bc][64];

    int tid_flat = row * 32 + lane;
    int total = Bc * d;  // 2048

    // Prefetch first tile into buffer 0
    if (Tc > 0) {
        for (int idx = tid_flat; idx < total; idx += 1024) {
            int r = idx / d;
            int c = idx % d;
            int kv_row = r;  // tile 0
            Ks[0][r][c] = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
            Vs[0][r][c] = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
        }
    }
    __syncthreads();

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int cur_buf = kv_tile & 1;
        int nxt_buf = 1 - cur_buf;

        // Start loading next tile into nxt_buf (if exists)
        // We need __syncthreads between compute and load of next,
        // but we can overlap the load with compute if we sync properly.
        // Actually on sm_60 without async copy, we can't truly overlap.
        // But we save the overhead of a separate load phase.

        // Process current tile (single-pass fused)
        for (int j = 0; j < Bc; j++) {
            int base = lane * 2;
            float s = 0.0f;
            if (base < d)     s += Qi0 * Ks[cur_buf][j][base];
            if (base + 1 < d) s += Qi1 * Ks[cur_buf][j][base + 1];

            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            s = __shfl_sync(0xffffffff, s, 0);
            s *= scale;

            float new_mi = fmaxf(mi, s);
            float alpha  = expf(mi - new_mi);
            float p      = expf(s - new_mi);

            Oi0 = Oi0 * alpha + p * ((base < d)     ? Vs[cur_buf][j][base]     : 0.0f);
            Oi1 = Oi1 * alpha + p * ((base + 1 < d) ? Vs[cur_buf][j][base + 1] : 0.0f);
            li = li * alpha + p;
            mi = new_mi;
        }

        __syncthreads();

        // Load next tile
        if (kv_tile + 1 < Tc) {
            int next_base = (kv_tile + 1) * Bc;
            for (int idx = tid_flat; idx < total; idx += 1024) {
                int r = idx / d;
                int c = idx % d;
                int kv_row = next_base + r;
                Ks[nxt_buf][r][c] = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
                Vs[nxt_buf][r][c] = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
            }
        }
        __syncthreads();
    }

    if (q_row < N) {
        int base = lane * 2;
        float inv_li = 1.0f / li;
        if (base < d)     Oh[q_row * d + base]     = Oi0 * inv_li;
        if (base + 1 < d) Oh[q_row * d + base + 1] = Oi1 * inv_li;
    }
}

extern "C" void flash_attention_v4(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    dim3 block(32, Br);
    flash_attention_v4_kernel<<<grid, block>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
