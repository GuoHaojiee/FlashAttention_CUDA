#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// Q, K, V, O: [bh, N, d]   S: [bh, N, N]

// Kernel 1: S[h][row][col] = dot(Q[h][row], K[h][col]) / sqrt(d)
// blockIdx.z = head, block (row, col) of S
// BUG NOTE: K access Kh[col*d+i] has stride d between warp threads -> non-coalesced.
// Fixing this properly requires tiling K to shared memory (becoming a GEMM).
// Left as-is since this is the intentionally naive baseline.
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
    float*       Sh = S + (size_t)h * N * N;

    float s = 0.0f;
    for (int i = 0; i < d; i++)
        s += Qh[row * d + i] * Kh[col * d + i];
    Sh[row * N + col] = s * scale;
}

// Kernel 2: in-place row softmax.
// One block per row (blockIdx.x = row, blockIdx.y = head).
// Threads within a block collaborate on one row -> coalesced access:
//   thread tid reads Sh[tid], Sh[tid+blockDim], ... (consecutive within warp)
// Previous design (one thread per row) had stride-N access between warp threads.
__global__ void softmax_kernel(float* S, int bh, int N)
{
    int h   = blockIdx.y;
    int row = blockIdx.x;
    if (h >= bh || row >= N) return;

    float* Sh  = S + (size_t)h * N * N + (size_t)row * N;
    int    tid = threadIdx.x;
    int    bdim = blockDim.x;

    __shared__ float smem[32];

    // Pass 1: parallel max reduction (coalesced: warp reads Sh[0..31], Sh[32..63], ...)
    float local_max = -1e9f;
    for (int j = tid; j < N; j += bdim)
        local_max = fmaxf(local_max, Sh[j]);

    // Warp reduce
    for (int mask = 16; mask > 0; mask >>= 1)
        local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, mask));
    if (tid % 32 == 0) smem[tid / 32] = local_max;
    __syncthreads();

    // Block reduce (first warp)
    if (tid < 32) {
        local_max = (tid < bdim / 32) ? smem[tid] : -1e9f;
        for (int mask = 16; mask > 0; mask >>= 1)
            local_max = fmaxf(local_max, __shfl_down_sync(0xffffffff, local_max, mask));
        if (tid == 0) smem[0] = local_max;
    }
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: parallel exp-sum
    float local_sum = 0.0f;
    for (int j = tid; j < N; j += bdim)
        local_sum += expf(Sh[j] - row_max);

    for (int mask = 16; mask > 0; mask >>= 1)
        local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
    if (tid % 32 == 0) smem[tid / 32] = local_sum;
    __syncthreads();

    if (tid < 32) {
        local_sum = (tid < bdim / 32) ? smem[tid] : 0.0f;
        for (int mask = 16; mask > 0; mask >>= 1)
            local_sum += __shfl_down_sync(0xffffffff, local_sum, mask);
        if (tid == 0) smem[0] = local_sum;
    }
    __syncthreads();
    float row_sum = smem[0];

    // Pass 3: write normalized values (coalesced write)
    for (int j = tid; j < N; j += bdim)
        Sh[j] = expf(Sh[j] - row_max) / row_sum;
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
    float*       Oh = O + (size_t)h * N * d;

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
        // One block per row, 128 threads collaborate on the row
        dim3 block(128);
        dim3 grid(N, bh);
        softmax_kernel<<<grid, block>>>(S, bh, N);
    }
    {
        dim3 block(16, 16);
        dim3 grid((d + 15) / 16, (N + 15) / 16, bh);
        compute_O_kernel<<<grid, block>>>(S, V, O, bh, N, d);
    }
    cudaDeviceSynchronize();
}
