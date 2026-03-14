/*
 * K2: FlashAttention + 寄存器存 l/m + float4 向量化加载
 *
 * 在 K1 基础上的两个优化：
 *
 * 优化1：l/m 移入寄存器
 *   K1 中 l/m 每次 kv_tile 循环后都从 HBM 读出/写回，
 *   实际上它们只在最后写回一次就够了，中间过程完全可以放寄存器。
 *   消除了 Tr × Tc 次不必要的 HBM 往返。
 *
 * 优化2：float4 向量化加载
 *   把 Q/K/V 从全局内存加载到 SRAM 时，改用 float4（一次读 16 bytes），
 *   更好地利用内存带宽（P100 的 L2 cache line = 128 bytes，float4 对齐有利于合并访问）。
 *   要求 d 是 4 的倍数。
 *
 * 编译：nvcc -O2 -arch=sm_60 flash_attention_v2.cu -o flash_attention_v2.so --shared -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define Br 32
#define Bc 32

__global__ void flash_attention_v2_kernel(
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

    // l/m 完全在寄存器中，不碰 HBM
    float mi = -1e9f;
    float li = 0.0f;

    __shared__ float Qs[Br][64];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];
    __shared__ float Ss[Br][Bc];

    float Oi[64] = {0.0f};
    float scale = 1.0f / sqrtf((float)d);

    // float4 向量化加载 Q tile（每次加载4个float，减少指令数）
    // 要求 d % 4 == 0
    int d4 = d / 4;
    if (q_row < N) {
        const float4* Q4 = reinterpret_cast<const float4*>(Q + q_row * d);
        float4* Qs4 = reinterpret_cast<float4*>(Qs[tid]);
        for (int i = 0; i < d4; i++) {
            Qs4[i] = Q4[i];
        }
    } else {
        for (int i = 0; i < d; i++) Qs[tid][i] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int kv_row = kv_tile * Bc + tid;

        // float4 向量化加载 K、V tile
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

        // 计算 S tile
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) {
                s += Qs[tid][i] * Ks[j][i];
            }
            Ss[tid][j] = s * scale;
        }
        __syncthreads();

        // Online Softmax：m 和 l 全程在寄存器，无 HBM 读写
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

    // 最后统一写回 O，l/m 不再写入 HBM
    if (q_row < N) {
        for (int i = 0; i < d; i++) {
            O[q_row * d + i] = Oi[i] / li;
        }
    }
}

extern "C" void flash_attention_v2(
    const float* Q, const float* K, const float* V,
    float* O,
    int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    flash_attention_v2_kernel<<<Tr, Bc>>>(Q, K, V, O, N, d);
    cudaDeviceSynchronize();
}
