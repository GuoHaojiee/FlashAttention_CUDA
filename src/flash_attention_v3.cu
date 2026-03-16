#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// K3: FlashAttention + Warp-parallel + Single-pass fused online softmax
//
// Improvement over K2:
//   K1/K2 use a TWO-PASS approach per KV tile:
//     Pass 1: compute all dot products to find new max
//     Pass 2: compute exp and accumulate O
//   This means each Ks[j][*] is read TWICE from shared memory.
//
//   K3 uses a SINGLE-PASS fused approach:
//     For each j, compute dot product, update running max, rescale, accumulate.
//     Each Ks[j][*] and Vs[j][*] read only ONCE.
//     Trade-off: one extra rescale per j, but saves 50% shared memory reads.
//
// Same warp-per-row architecture: blockDim = (32, Br) = 1024 threads.

__global__ __launch_bounds__(1024, 2)
void flash_attention_v3_kernel(
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

    int lane  = threadIdx.x;  // 0..31
    int row   = threadIdx.y;  // 0..Br-1
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

    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        // Cooperative load
        {
            int tid_flat = row * 32 + lane;
            int total = Bc * d;
            for (int idx = tid_flat; idx < total; idx += 1024) {
                int r = idx / d;
                int c = idx % d;
                int kv_row = kv_tile * Bc + r;
                Ks[r][c] = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
                Vs[r][c] = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
            }
        }
        __syncthreads();

        // Single-pass fused online softmax: process one j at a time
        for (int j = 0; j < Bc; j++) {
            int base = lane * 2;
            float s = 0.0f;
            if (base < d)     s += Qi0 * Ks[j][base];
            if (base + 1 < d) s += Qi1 * Ks[j][base + 1];

            // Warp reduce
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            s = __shfl_sync(0xffffffff, s, 0);
            s *= scale;

            // Online softmax update
            float new_mi = fmaxf(mi, s);
            float alpha  = expf(mi - new_mi);
            float p      = expf(s - new_mi);

            // Rescale running accumulator and add new contribution
            Oi0 = Oi0 * alpha + p * ((base < d)     ? Vs[j][base]     : 0.0f);
            Oi1 = Oi1 * alpha + p * ((base + 1 < d) ? Vs[j][base + 1] : 0.0f);
            li = li * alpha + p;
            mi = new_mi;
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

extern "C" void flash_attention_v3(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    dim3 block(32, Br);
    flash_attention_v3_kernel<<<grid, block>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
