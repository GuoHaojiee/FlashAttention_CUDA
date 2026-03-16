#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// K2: FlashAttention + Warp-parallel + float4 vectorized loads
//
// Same warp-per-row architecture as K1, plus:
//   - float4 vectorized global memory loads for Q, K, V, O
//   - Each lane handles d/32=2 elements, loaded as individual floats
//     (float4 used for cooperative K/V tile loading)
//
// Occupancy: same as K1 — blockDim=1024, 2 blocks/SM, 100% occupancy
//
// Improvement over K1: faster K/V tile loading via float4

__global__ __launch_bounds__(1024, 2)
void flash_attention_v2_kernel(
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
        // Cooperative load using float4: 1024 threads load 2048 floats = 512 float4s
        // Each thread loads at most 1 float4 (4 floats)
        {
            int tid_flat = row * 32 + lane;
            int total4 = (Bc * d) / 4;  // 512 float4s
            // Load K tile
            for (int idx4 = tid_flat; idx4 < total4; idx4 += 1024) {
                int elem = idx4 * 4;
                int r = elem / d;
                int c = elem % d;
                int kv_row = kv_tile * Bc + r;
                float4 val;
                if (kv_row < N) {
                    val = reinterpret_cast<const float4*>(Kh + kv_row * d)[c / 4];
                } else {
                    val.x = val.y = val.z = val.w = 0.0f;
                }
                reinterpret_cast<float4*>(&Ks[r][c])[0] = val;
            }
            // Load V tile
            for (int idx4 = tid_flat; idx4 < total4; idx4 += 1024) {
                int elem = idx4 * 4;
                int r = elem / d;
                int c = elem % d;
                int kv_row = kv_tile * Bc + r;
                float4 val;
                if (kv_row < N) {
                    val = reinterpret_cast<const float4*>(Vh + kv_row * d)[c / 4];
                } else {
                    val.x = val.y = val.z = val.w = 0.0f;
                }
                reinterpret_cast<float4*>(&Vs[r][c])[0] = val;
            }
        }
        __syncthreads();

        float new_mi = mi;

        // First pass: find new max
        for (int j = 0; j < Bc; j++) {
            int base = lane * 2;
            float s = 0.0f;
            if (base < d)     s += Qi0 * Ks[j][base];
            if (base + 1 < d) s += Qi1 * Ks[j][base + 1];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            s = __shfl_sync(0xffffffff, s, 0);
            new_mi = fmaxf(new_mi, s * scale);
        }

        float alpha = expf(mi - new_mi);
        Oi0 *= alpha;
        Oi1 *= alpha;
        li *= alpha;

        // Second pass: accumulate
        for (int j = 0; j < Bc; j++) {
            int base = lane * 2;
            float s = 0.0f;
            if (base < d)     s += Qi0 * Ks[j][base];
            if (base + 1 < d) s += Qi1 * Ks[j][base + 1];
            for (int offset = 16; offset > 0; offset >>= 1)
                s += __shfl_down_sync(0xffffffff, s, offset);
            s = __shfl_sync(0xffffffff, s, 0);
            float p = expf(s * scale - new_mi);
            li += p;
            if (base < d)     Oi0 += p * Vs[j][base];
            if (base + 1 < d) Oi1 += p * Vs[j][base + 1];
        }

        mi = new_mi;
        __syncthreads();
    }

    if (q_row < N) {
        int base = lane * 2;
        float inv_li = 1.0f / li;
        if (base < d)     Oh[q_row * d + base]     = Oi0 * inv_li;
        if (base + 1 < d) Oh[q_row * d + base + 1] = Oi1 * inv_li;
    }
}

extern "C" void flash_attention_v2(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    dim3 block(32, Br);
    flash_attention_v2_kernel<<<grid, block>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
