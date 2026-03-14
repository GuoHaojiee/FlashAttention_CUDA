/*
 * K0: Naive Attention — Memory Bound Baseline
 *
 * 实现 O = softmax(QK^T / sqrt(d)) * V
 * 瓶颈：S 矩阵大小为 N×N，必须完整写入再读出 HBM，
 *       DRAM 访问量随 N² 增长，严重 Memory Bound。
 *
 * 编译：nvcc -O2 -arch=sm_60 naive_attention.cu -o naive_attention.so --shared -Xcompiler -fPIC
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Kernel 1: 计算 S = QK^T / sqrt(d)，每个线程负责 S 的一个元素
__global__ void compute_S_kernel(
    const float* Q, const float* K, float* S,
    int N, int d, float scale)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    float s = 0.0f;
    for (int i = 0; i < d; i++) {
        s += Q[row * d + i] * K[col * d + i];
    }
    S[row * N + col] = s * scale;
}

// Kernel 2: 对 S 做行 softmax，结果存回 S（in-place）
// 每个线程处理一整行
__global__ void softmax_kernel(float* S, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // 找行最大值（数值稳定）
    float m = -1e9f;
    for (int j = 0; j < N; j++) {
        m = fmaxf(m, S[row * N + j]);
    }

    // 计算归一化因子
    float l = 0.0f;
    for (int j = 0; j < N; j++) {
        l += expf(S[row * N + j] - m);
    }

    // 写回 softmax 结果
    for (int j = 0; j < N; j++) {
        S[row * N + j] = expf(S[row * N + j] - m) / l;
    }
}

// Kernel 3: O = P * V，每个线程负责 O 的一个元素
// P = softmax(S) 已存在 S 中
__global__ void compute_O_kernel(
    const float* S, const float* V, float* O,
    int N, int d)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= d) return;

    float o = 0.0f;
    for (int j = 0; j < N; j++) {
        o += S[row * N + j] * V[j * d + col];
    }
    O[row * d + col] = o;
}

extern "C" void naive_attention(
    const float* Q, const float* K, const float* V,
    float* S, float* O,
    int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    // Step 1: S = QK^T / sqrt(d)，写入全局内存（HBM）
    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16);
        compute_S_kernel<<<grid, block>>>(Q, K, S, N, d, scale);
    }

    // Step 2: 行 softmax，S 从 HBM 读出再写回 HBM（两次 DRAM 访问）
    {
        dim3 block(256);
        dim3 grid((N + 255) / 256);
        softmax_kernel<<<grid, block>>>(S, N);
    }

    // Step 3: O = softmax(S) * V，再次从 HBM 读 S
    {
        dim3 block(16, 16);
        dim3 grid((d + 15) / 16, (N + 15) / 16);
        compute_O_kernel<<<grid, block>>>(S, V, O, N, d);
    }

    cudaDeviceSynchronize();
}
