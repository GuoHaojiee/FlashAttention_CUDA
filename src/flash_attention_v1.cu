#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// Q, K, V, O: [bh, N, d]   l, m: [bh, N]
// grid = (Tr, bh): blockIdx.x = Q tile, blockIdx.y = head index
// 每个 block 处理一个 head 的一个 Q tile，bh 个 head 完全并行

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

    // 当前 head 在全局数组中的偏移
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

    float Oi[64] = {0.0f};
    float scale  = 1.0f / sqrtf((float)d);

    if (q_row < N) {
        for (int i = 0; i < d; i++) Qs[tid][i] = Qh[q_row * d + i];
    } else {
        for (int i = 0; i < d; i++) Qs[tid][i] = 0.0f;
    }

    // l/m 初始化写入 HBM（K1 特征：每次 kv_tile 都读写 HBM 中的 l/m）
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

        // 从 HBM 读取 m 和 l（与 K2 的对比点）
        float mi = (q_row < N) ? mh[q_row] : -1e9f;
        float li = (q_row < N) ? lh[q_row] : 0.0f;

        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, Ss[tid][j]);

        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) lj += expf(Ss[tid][j] - mj);

        float alpha  = expf(mi - mj);
        float li_new = li * alpha + lj;

        for (int i = 0; i < d; i++) {
            float vi_sum = 0.0f;
            for (int j = 0; j < Bc; j++) vi_sum += expf(Ss[tid][j] - mj) * Vs[j][i];
            Oi[i] = Oi[i] * alpha + vi_sum;
        }

        // 将更新后的 m/l 写回 HBM
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
    flash_attention_v1_kernel<<<grid, Bc>>>(Q, K, V, O, l, m, bh, N, d);
    cudaDeviceSynchronize();
}
