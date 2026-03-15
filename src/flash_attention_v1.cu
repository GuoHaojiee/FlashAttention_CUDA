#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// Q, K, V, O: [bh, N, d]   l, m: [bh, N]
// grid = (Tr, bh): blockIdx.x = Q tile, blockIdx.y = head index

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

    int tid   = threadIdx.x;
    int q_row = q_tile * Br + tid;

    __shared__ float Qs[Br][64];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];
    __shared__ float Ss[Br][Bc];

    // Oi lives entirely in registers: no HBM access inside the kv_tile loop
    float Oi[64] = {0.0f};
    float scale  = 1.0f / sqrtf((float)d);

    if (q_row < N) {
        for (int i = 0; i < d; i++) Qs[tid][i] = Qh[q_row * d + i];
    } else {
        for (int i = 0; i < d; i++) Qs[tid][i] = 0.0f;
    }

    // K1: l/m live in HBM and are read/written once per kv_tile (vs K2 where they stay in registers)
    if (q_row < N) { lh[q_row] = 0.0f; mh[q_row] = -1e9f; }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int kv_row = kv_tile * Bc + tid;

        if (kv_row < N) {
            for (int i = 0; i < d; i++) {
                Ks[tid][i] = Kh[kv_row * d + i];
                Vs[tid][i] = Vh[kv_row * d + i];
            }
        } else {
            for (int i = 0; i < d; i++) { Ks[tid][i] = 0.0f; Vs[tid][i] = 0.0f; }
        }
        __syncthreads();

        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += Qs[tid][i] * Ks[j][i];
            Ss[tid][j] = s * scale;
        }
        __syncthreads();

        // K1: read l/m from HBM each kv_tile
        float mi = (q_row < N) ? mh[q_row] : -1e9f;
        float li = (q_row < N) ? lh[q_row] : 0.0f;

        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, Ss[tid][j]);

        // Precompute exp once per j -- previously expf was called (d+1) times per j,
        // totaling (d+1)*Bc = 65*32 = 2080 expf/kv_tile. Now it's Bc = 32.
        float exp_s[Bc];
        for (int j = 0; j < Bc; j++) exp_s[j] = expf(Ss[tid][j] - mj);

        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) lj += exp_s[j];

        float alpha  = expf(mi - mj);
        float li_new = li * alpha + lj;

        // Rescale accumulated O, then add this tile's contribution.
        // j-outer loop: accesses Vs[j][0..d-1] sequentially (good smem locality).
        // Previously i-outer accessed Vs with stride d between j steps.
        for (int i = 0; i < d; i++) Oi[i] *= alpha;
        for (int j = 0; j < Bc; j++) {
            float esj = exp_s[j];
            for (int i = 0; i < d; i++) Oi[i] += esj * Vs[j][i];
        }

        // K1: write l/m back to HBM each kv_tile
        if (q_row < N) { mh[q_row] = mj; lh[q_row] = li_new; }
        mi = mj; li = li_new;
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
    flash_attention_v1_kernel<<<grid, Bc>>>(Q, K, V, O, l, m, bh, N, d);
    cudaDeviceSynchronize();
}
