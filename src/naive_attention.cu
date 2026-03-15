#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Q, K, V, O: [bh, N, d]   S: [bh, N, N]
// bh = batch * num_heads，每个 head 独立做一次 attention

// Kernel 1: S[h][row][col] = dot(Q[h][row], K[h][col]) / sqrt(d)
// blockIdx.z = head index, 使 bh 个 head 完全并行
__global__ void compute_S_kernel(
    const float* Q, const float* K, float* S,
    int bh, int N, int d, float scale)
{
    int h   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= bh || row >= N || col >= N) return;

    const float* Qh = Q + (size_t)h * N * d;
    const float* Kh = K + (size_t)h * N * d;
    float* Sh = S + (size_t)h * N * N;

    float s = 0.0f;
    for (int i = 0; i < d; i++)
        s += Qh[row * d + i] * Kh[col * d + i];
    Sh[row * N + col] = s * scale;
}

// Kernel 2: in-place row softmax on S[h]
__global__ void softmax_kernel(float* S, int bh, int N)
{
    int h   = blockIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= bh || row >= N) return;

    float* Sh = S + (size_t)h * N * N;
    float m = -1e9f;
    for (int j = 0; j < N; j++) m = fmaxf(m, Sh[row * N + j]);

    float l = 0.0f;
    for (int j = 0; j < N; j++) l += expf(Sh[row * N + j] - m);

    for (int j = 0; j < N; j++)
        Sh[row * N + j] = expf(Sh[row * N + j] - m) / l;
}

// Kernel 3: O[h] = softmax(S[h]) * V[h], one thread per output element
__global__ void compute_O_kernel(
    const float* S, const float* V, float* O,
    int bh, int N, int d)
{
    int h   = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= bh || row >= N || col >= d) return;

    const float* Sh = S + (size_t)h * N * N;
    const float* Vh = V + (size_t)h * N * d;
    float* Oh = O + (size_t)h * N * d;

    float o = 0.0f;
    for (int j = 0; j < N; j++)
        o += Sh[row * N + j] * Vh[j * d + col];
    Oh[row * d + col] = o;
}

extern "C" void naive_attention(
    const float* Q, const float* K, const float* V,
    float* S, float* O,
    int bh, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    {
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (N + 15) / 16, bh);
        compute_S_kernel<<<grid, block>>>(Q, K, S, bh, N, d, scale);
    }
    {
        dim3 block(256);
        dim3 grid((N + 255) / 256, bh);
        softmax_kernel<<<grid, block>>>(S, bh, N);
    }
    {
        dim3 block(16, 16);
        dim3 grid((d + 15) / 16, (N + 15) / 16, bh);
        compute_O_kernel<<<grid, block>>>(S, V, O, bh, N, d);
    }
    cudaDeviceSynchronize();
}
