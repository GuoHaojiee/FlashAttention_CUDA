/*
 * K3: FlashAttention + 消除 Shared Memory Bank Conflict
 *
 * 在 K2 基础上的优化：
 *
 * 问题：Ss[Br][Bc] 大小为 32×32，读取 Ss[tid][j] 时，
 *       32 个线程分别访问 Ss[0][j]~Ss[31][j]（同列不同行），
 *       每行 32 个 float = 128 bytes，恰好跨 4 个 bank（每 bank 4 bytes），
 *       列访问时 Ss[i][j] 对应 bank = (i*32 + j) % 32，
 *       当 j 固定，i=0..31 时 bank = (i*32+j)%32 = j，全部同一 bank → bank conflict！
 *
 * 解决：每行末尾 padding 1 个 float（Ss[Br][Bc+1]），
 *       使每行占 33 个 float，打破 32-bank 的对齐规律：
 *       bank = (i*33 + j) % 32，不同 i 对应不同 bank，消除冲突。
 *
 * 编译：nvcc -O2 -arch=sm_60 flash_attention_v3.cu -o flash_attention_v3.so --shared -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define Br 32
#define Bc 32
#define PADDING 1   // 每行末尾 padding 的 float 数，打破 bank 对齐

__global__ void flash_attention_v3_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int N, int d)
{
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;

    int q_tile = blockIdx.x;
    if (q_tile >= Tr) return;

    int tid = threadIdx.x;
    int q_row = q_tile * Br + tid;

    float mi = -1e9f;
    float li = 0.0f;

    __shared__ float Qs[Br][64];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];
    // Ss 加 padding：每行多一个 float，使列访问无 bank conflict
    __shared__ float Ss[Br][Bc + PADDING];

    float Oi[64] = {0.0f};
    float scale = 1.0f / sqrtf((float)d);

    int d4 = d / 4;
    if (q_row < N) {
        const float4* Q4 = reinterpret_cast<const float4*>(Q + q_row * d);
        float4* Qs4 = reinterpret_cast<float4*>(Qs[tid]);
        for (int i = 0; i < d4; i++) Qs4[i] = Q4[i];
    } else {
        for (int i = 0; i < d; i++) Qs[tid][i] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int kv_row = kv_tile * Bc + tid;

        if (kv_row < N) {
            const float4* K4 = reinterpret_cast<const float4*>(K + kv_row * d);
            const float4* V4 = reinterpret_cast<const float4*>(V + kv_row * d);
            float4* Ks4 = reinterpret_cast<float4*>(Ks[tid]);
            float4* Vs4 = reinterpret_cast<float4*>(Vs[tid]);
            for (int i = 0; i < d4; i++) {
                Ks4[i] = K4[i];
                Vs4[i] = V4[i];
            }
        } else {
            for (int i = 0; i < d; i++) { Ks[tid][i] = 0.0f; Vs[tid][i] = 0.0f; }
        }
        __syncthreads();

        // 写入带 padding 的 Ss：Ss[tid][j]，访问模式无 bank conflict
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) {
                s += Qs[tid][i] * Ks[j][i];
            }
            Ss[tid][j] = s * scale;  // 写时每行步长 Bc+PADDING，bank 错开
        }
        __syncthreads();

        // 读取 Ss 时，32 个线程访问同列 Ss[0..31][j]
        // 由于步长=33，bank = (i*33+j)%32 各不相同，无 conflict
        float mj = mi;
        for (int j = 0; j < Bc; j++) {
            mj = fmaxf(mj, Ss[tid][j]);
        }

        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) {
            lj += expf(Ss[tid][j] - mj);
        }

        float alpha = expf(mi - mj);
        float li_new = li * alpha + lj;

        for (int i = 0; i < d; i++) {
            float vi_sum = 0.0f;
            for (int j = 0; j < Bc; j++) {
                vi_sum += expf(Ss[tid][j] - mj) * Vs[j][i];
            }
            Oi[i] = Oi[i] * alpha + vi_sum;
        }

        mi = mj;
        li = li_new;

        __syncthreads();
    }

    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            O[q_row * d + i] = Oi[i] / li;
        }
    }
}

extern "C" void flash_attention_v3(
    const float* Q, const float* K, const float* V,
    float* O,
    int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    flash_attention_v3_kernel<<<Tr, Bc>>>(Q, K, V, O, N, d);
    cudaDeviceSynchronize();
}
