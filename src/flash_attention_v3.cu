#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32
#define PADDING 1  // Ss 每行 padding 1 个 float，消除列访问的 bank conflict

// Ss[Br][Bc+PADDING]: 步长从 32 变为 33
//   列访问时 bank = (row*33 + col) % 32，不同 row 映射不同 bank，无 conflict
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

    __shared__ float Qs[Br][64];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];
    __shared__ float Ss[Br][Bc + PADDING];  // padding 消除 bank conflict

    float Oi[64] = {0.0f};
    float scale  = 1.0f / sqrtf((float)d);
    int d4 = d / 4;

    if (q_row < N) {
        const float4* Q4 = reinterpret_cast<const float4*>(Qh + q_row * d);
        float4* Qs4 = reinterpret_cast<float4*>(Qs[tid]);
        for (int i = 0; i < d4; i++) Qs4[i] = Q4[i];
    } else {
        for (int i = 0; i < d; i++) Qs[tid][i] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
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
        __syncthreads();

        // 写 Ss：行步长 = Bc+PADDING，bank = (tid*(Bc+1)+j)%32 各不同
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += Qs[tid][i] * Ks[j][i];
            Ss[tid][j] = s * scale;
        }
        __syncthreads();

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
