#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// K1: Basic FlashAttention — Tiling + Online Softmax + Warp-parallel dot products
//
// Key architecture change vs old K1:
//   OLD: blockDim = (Br), 1 thread per Q row, serial d=64 dot product → 0.2 GB/s
//   NEW: blockDim = (32, Br), 32 threads (1 warp) per Q row, parallel dot product
//
// Each warp handles one Q row:
//   - lane = threadIdx.x (0..31): which slice of d dimension
//   - row  = threadIdx.y (0..Br-1): which Q row within the tile
//   - Each lane loads d/32 = 2 elements of Qi, computes partial dot products,
//     then warp-reduces via __shfl_down_sync
//
// Shared memory layout:
//   Ks[Bc][d] = 32×64×4 = 8KB
//   Vs[Bc][d] = 32×64×4 = 8KB
//   Total = 16KB → 48KB/16KB = 3 blocks/SM
//
// Occupancy analysis (P100, sm_60):
//   blockDim = 32×32 = 1024 threads
//   Registers: each thread holds ~15 values (2 Qi + 2 Oi + loop vars + ss partial)
//     → ~20 regs/thread
//   65536/(20×1024) = 3.2 → 3 blocks from regs
//   48KB/16KB = 3 blocks from shared
//   2048/1024 = 2 blocks from max threads/SM → 2 blocks/SM
//   2×1024 = 2048 threads/SM = 64 warps = 100% occupancy!

__global__ __launch_bounds__(1024, 2)
void flash_attention_v1_kernel(
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

    int lane  = threadIdx.x;  // 0..31 (within warp)
    int row   = threadIdx.y;  // 0..Br-1 (which Q row)
    int q_row = q_tile * Br + row;

    float scale = 1.0f / sqrtf((float)d);

    // Online softmax state — per row, but all lanes compute identical values
    float mi = -1e9f;
    float li = 0.0f;

    // Each lane accumulates its slice of the output
    // For d=64, each lane handles 2 elements: lane*2 and lane*2+1
    float Oi0 = 0.0f, Oi1 = 0.0f;

    // Load Qi for this lane's slice
    float Qi0 = 0.0f, Qi1 = 0.0f;
    if (q_row < N) {
        int base = lane * 2;
        if (base < d)     Qi0 = Qh[q_row * d + base];
        if (base + 1 < d) Qi1 = Qh[q_row * d + base + 1];
    }

    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        // Cooperative load K/V tile: 1024 threads load Bc*d = 2048 values
        // → 2 values per thread
        {
            int total = Bc * d;  // 2048
            int tid_flat = row * 32 + lane;  // 0..1023
            for (int idx = tid_flat; idx < total; idx += 1024) {
                int r = idx / d;   // which KV row within tile
                int c = idx % d;   // which d column
                int kv_row = kv_tile * Bc + r;
                Ks[r][c] = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
                Vs[r][c] = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
            }
        }
        __syncthreads();

        // For each KV row j in this tile: compute dot product, online softmax, accumulate O
        // Process one j at a time to minimize register pressure
        float new_mi = mi;

        // First pass: compute all scores and find new max
        // We process scores one at a time to avoid needing ss[Bc] array
        for (int j = 0; j < Bc; j++) {
            // Parallel dot product: each lane computes partial sum
            int base = lane * 2;
            float s = 0.0f;
            if (base < d)     s += Qi0 * Ks[j][base];
            if (base + 1 < d) s += Qi1 * Ks[j][base + 1];

            // Warp reduce sum
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            // Broadcast from lane 0 to all lanes
            s = __shfl_sync(0xffffffff, s, 0);
            s *= scale;

            new_mi = fmaxf(new_mi, s);
        }

        // Rescale old accumulator
        float alpha = expf(mi - new_mi);
        Oi0 *= alpha;
        Oi1 *= alpha;
        li *= alpha;

        // Second pass: compute exp(s - new_mi) and accumulate
        for (int j = 0; j < Bc; j++) {
            int base = lane * 2;
            float s = 0.0f;
            if (base < d)     s += Qi0 * Ks[j][base];
            if (base + 1 < d) s += Qi1 * Ks[j][base + 1];

            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            s = __shfl_sync(0xffffffff, s, 0);
            s *= scale;

            float p = expf(s - new_mi);
            li += p;

            // Accumulate O: each lane updates its own d-slice
            if (base < d)     Oi0 += p * Vs[j][base];
            if (base + 1 < d) Oi1 += p * Vs[j][base + 1];
        }

        mi = new_mi;
        __syncthreads();
    }

    // Write output
    if (q_row < N) {
        int base = lane * 2;
        float inv_li = 1.0f / li;
        if (base < d)     Oh[q_row * d + base]     = Oi0 * inv_li;
        if (base + 1 < d) Oh[q_row * d + base + 1] = Oi1 * inv_li;
    }
}

extern "C" void flash_attention_v1(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    dim3 block(32, Br);  // 32 lanes × Br rows = 1024 threads
    flash_attention_v1_kernel<<<grid, block>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
