/*
 * K1: Basic FlashAttention — Tiling + Online Softmax
 *
 * 核心思路（FlashAttention Algorithm 1）：
 *   不分配 N×N 的 S 矩阵，而是将 Q 分成 Tr 个 tile，K/V 分成 Tc 个 tile，
 *   tile-by-tile 地在 SRAM 中计算，用 Online Softmax 维护 m（行最大值）和 l（归一化因子）。
 *
 * 与 K2 的关键区别：
 *   l 和 m 每次 kv_tile 循环都从 HBM 读出再写回（显式全局内存访问），
 *   这与 tspeterkim/flash-attention-minimal 的实现方式相同，作为对照基准。
 *
 * 参考：https://github.com/tspeterkim/flash-attention-minimal
 *
 * 编译：nvcc -O2 -arch=sm_60 flash_attention_v1.cu -o flash_attention_v1.so --shared -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

#define Br 32
#define Bc 32

__global__ void flash_attention_v1_kernel(
    const float* Q,   // [N, d]
    const float* K,   // [N, d]
    const float* V,   // [N, d]
    float* O,         // [N, d]
    float* l,         // [N]  归一化因子，存在 HBM，每次 kv_tile 都读写
    float* m,         // [N]  行最大值，存在 HBM，每次 kv_tile 都读写
    int N, int d)
{
    int Tc = (N + Bc - 1) / Bc;
    int Tr = (N + Br - 1) / Br;

    int q_tile = blockIdx.x;
    if (q_tile >= Tr) return;

    int tid = threadIdx.x;
    int q_row = q_tile * Br + tid;  // 当前线程对应的全局行号

    __shared__ float Qs[Br][64];
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];
    __shared__ float Ss[Br][Bc];

    // 累加器在寄存器中
    float Oi[64] = {0.0f};

    float scale = 1.0f / sqrtf((float)d);

    // 初始化 HBM 中的 l 和 m
    if (q_row < N) {
        l[q_row] = 0.0f;
        m[q_row] = -1e9f;
    }

    // 加载 Q tile 到 SRAM（只加载一次，Q tile 在整个 kv 循环中不变）
    if (q_row < N) {
        for (int i = 0; i < d; i++) Qs[tid][i] = Q[q_row * d + i];
    } else {
        for (int i = 0; i < d; i++) Qs[tid][i] = 0.0f;
    }

    // 外层循环：遍历所有 K/V tile
    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int kv_row = kv_tile * Bc + tid;

        // 加载 K、V tile
        if (kv_row < N) {
            for (int i = 0; i < d; i++) {
                Ks[tid][i] = K[kv_row * d + i];
                Vs[tid][i] = V[kv_row * d + i];
            }
        } else {
            for (int i = 0; i < d; i++) { Ks[tid][i] = 0.0f; Vs[tid][i] = 0.0f; }
        }
        __syncthreads();

        // 计算 S tile = Q tile × K tile^T / sqrt(d)
        for (int j = 0; j < Bc; j++) {
            float s = 0.0f;
            for (int i = 0; i < d; i++) s += Qs[tid][i] * Ks[j][i];
            Ss[tid][j] = s * scale;
        }
        __syncthreads();

        // 从 HBM 读取当前的 m 和 l（K1 特征：每次 kv_tile 都做 HBM 读写）
        float mi = (q_row < N) ? m[q_row] : -1e9f;
        float li = (q_row < N) ? l[q_row] : 0.0f;

        // Online Softmax：更新 m
        float mj = mi;
        for (int j = 0; j < Bc; j++) mj = fmaxf(mj, Ss[tid][j]);

        // 当前 tile 的 exp sum
        float lj = 0.0f;
        for (int j = 0; j < Bc; j++) lj += expf(Ss[tid][j] - mj);

        // 修正历史 O 累加值（旧 m → 新 m 的缩放）
        float alpha = expf(mi - mj);
        float li_new = li * alpha + lj;

        for (int i = 0; i < d; i++) {
            float vi_sum = 0.0f;
            for (int j = 0; j < Bc; j++) vi_sum += expf(Ss[tid][j] - mj) * Vs[j][i];
            Oi[i] = Oi[i] * alpha + vi_sum;
        }

        // 将更新后的 m 和 l 写回 HBM（K1 vs K2 的核心区别）
        if (q_row < N) {
            m[q_row] = mj;
            l[q_row] = li_new;
        }

        __syncthreads();
    }

    // 写回 O = Oi / l
    if (q_row < N) {
        float li_final = l[q_row];
        for (int i = 0; i < d; i++) {
            O[q_row * d + i] = Oi[i] / li_final;
        }
    }
}

extern "C" void flash_attention_v1(
    const float* Q, const float* K, const float* V,
    float* O, float* l, float* m,
    int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    flash_attention_v1_kernel<<<Tr, Bc>>>(Q, K, V, O, l, m, N, d);
    cudaDeviceSynchronize();
}
