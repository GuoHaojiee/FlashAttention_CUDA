#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// Optimization over K2: eliminate 32-way bank conflicts via smem transposition.
//
// Root cause in K2:
//   Qs[Br][64] accessed as Qs[tid][i]:
//     address = tid*64 + i,  bank = (tid*64 + i) % 32 = i % 32  (64%32=0)
//     -> ALL 32 threads hit bank i%32 -> 32-way conflict on every dot-product read.
//   Ss[Br][Bc] accessed as Ss[tid][j]:
//     address = tid*32 + j,  bank = (tid*32 + j) % 32 = j % 32
//     -> ALL 32 threads hit bank j%32 -> 32-way conflict.
//
// Fix: transpose Qs and Ss.
//   Qs[64][Br] accessed as Qs[i][tid]:
//     address = i*32 + tid,  bank = (i*32 + tid) % 32 = tid
//     -> thread t hits bank t -> 32 distinct banks -> conflict-free.
//   Ss[Bc][Br] accessed as Ss[j][tid]:
//     address = j*32 + tid,  bank = tid -> conflict-free.
//   Ks[Bc][64] and Vs[Bc][64] accessed as Ks[j][i] / Vs[j][i]:
//     all threads read the SAME (j,i) -> broadcast -> no conflict (unchanged).
//
// grid = (Tr, bh)

__global__ void flash_attention_v3_kernel(
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

    // Transposed layouts:
    //   Qs[d][Br]  = Qs[64][32]: Qs[i][tid] -> bank=tid -> conflict-free
    //   Ks[Bc][64] = Ks[32][64]: Ks[j][i]  -> broadcast  -> conflict-free
    //   Vs[Bc][64] = Vs[32][64]: Vs[j][i]  -> broadcast  -> conflict-free
    //   Ss[Bc][Br] = Ss[32][32]: Ss[j][tid] -> bank=tid  -> conflict-free
    __shared__ float Qs[64][Br];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];
    __shared__ float Ss[Bc][Br];

    float Oi[64] = {0.0f};
    float scale  = 1.0f / sqrtf((float)d);
    int d4 = d / 4;

    // Load Q: float4 from HBM, scatter to transposed smem (Qs[i][tid]).
    // The float4 read is coalesced within each thread's row; the smem write
    // goes to bank=tid for all four elements -> conflict-free.
    if (q_row < N) {
        for (int i = 0; i < d; i += 4) {
            float4 q4 = reinterpret_cast<const float4*>(Qh + q_row * d + i)[0];
            Qs[i+0][tid] = q4.x;
            Qs[i+1][tid] = q4.y;
            Qs[i+2][tid] = q4.z;
            Qs[i+3][tid] = q4.w;
        }
    } else {
        for (int i = 0; i < d; i++) Qs[i][tid] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int kv_row = kv_tile * Bc + tid;

        // K/V layout unchanged (Ks[tid][...] = row kv_row); broadcast access is fine.
        if (kv_row < N) {
            const float4* K4 = reinterpret_cast<const float4*>(Kh + kv_row * d);
            const float4* V4 = reinterpret_cast<const float4*>(Vh + kv_row * d);
            float4* Ks4 = reinterpret_cast<float4*>(Ks[tid]);
            float4* Vs4 = reinterpret_cast<float4*>(Vs[tid]);
            for (int i = 0; i < d4; i++) { Ks4[i] = K4[i]; Vs4[i] = V4[i]; }
        } else {
            for (int i = 0; i < d; i++) { Ks[tid][i] = 0.0f; Vs[tid][i] = 0.0f; }
        }
        __syncthreads();

        // Dot product: Qs[i][tid] (bank=tid, no conflict), Ks[j][i] (broadcast).
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += Qs[i][tid] * Ks[j][i];
            Ss[j][tid] = s * scale;  // bank=tid, no conflict
        }
        __syncthreads();

        // Ss[j][tid]: bank=tid -> conflict-free reads below.
        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, Ss[j][tid]);

        float exp_s[Bc];
        for (int j = 0; j < Bc; j++) exp_s[j] = expf(Ss[j][tid] - mj);

        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) lj += exp_s[j];

        float alpha  = expf(mi - mj);
        float li_new = li * alpha + lj;

        for (int i = 0; i < d; i++) Oi[i] *= alpha;
        for (int j = 0; j < Bc; j++) {
            float esj = exp_s[j];
            for (int i = 0; i < d; i++) Oi[i] += esj * Vs[j][i];  // broadcast
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
    flash_attention_v3_kernel<<<grid, Bc>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
