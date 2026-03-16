#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// K6: GEMM-style thread mapping — eliminates 30 of 32 warp reductions per tile
//
// Root cause of K1-K5 bottleneck:
//   K1-K5: thread (lane, row) holds 2 elements of Q. To compute one dot product
//   Q[row]·K[j], all 32 lanes reduce their partial sums via 5 __shfl_down_sync.
//   32 K columns per tile = 32 serial warp reductions = 160 dependent shuffles.
//   At ~5 cycles/shuffle: 800 cycles of pure reduction overhead per tile.
//
// K6 solution: GEMM-style 2D thread assignment.
//   tx = threadIdx.x  →  K column index (0..Bc-1)
//   ty = threadIdx.y  →  Q row index    (0..Br-1)
//
//   Each thread independently computes S[ty][tx] = Σ_i Q[ty][i] * K[tx][i]
//     → 64 FMAs, ZERO warp reduction for the dot product!
//
//   Warp reductions only happen for the softmax:
//     · 1 reduction to find row max (for numerical stability)
//     · 1 reduction to compute row sum (for normalizer)
//     = 2 reductions per tile  vs  32 in K1-K5
//
// Shared memory (28.8 KB < 48 KB P100 limit):
//   Qs[Br][64]   =  8 KB  Q tile, loaded once and reused across all Tc KV-tiles
//   Ks[Bc][65]   =  8.3KB K tile, +1 col padding → bank of Ks[tx][i] = (tx+i)%32
//                          → all 32 warp threads hit different banks, zero conflict
//   Vs[Bc][64]   =  8 KB  V tile
//   Ps[Br][Bc]   =  4 KB  attention weights p[ty][tx] for Phase 3 V accumulation
//
// Register pressure: ~20 regs/thread (no s[32] array) → low spill risk.
// Occupancy: 1 block/SM (28.8KB shmem; 2×28.8=57.6KB > 48KB), 50% like K4.
// Unlike K4, the reduced reduction overhead should yield much higher compute util.

__global__ __launch_bounds__(1024, 1)
void flash_attention_v6_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
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
    float*       Oh = O + (size_t)bh_idx * N * d;

    // tx: which K column (0..Bc-1); also d-dimension index in Phase 3 (d-elem pair tx*2, tx*2+1)
    // ty: which Q row within this tile (0..Br-1)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float scale = 1.0f / sqrtf((float)d);

    // Shared memory (total ~28.8 KB)
    __shared__ float Qs[Br][64];      // Q tile: stays for entire KV-tile loop
    __shared__ float Ks[Bc][65];      // K tile: +1 padding column to break bank conflicts
    __shared__ float Vs[Bc][64];      // V tile
    __shared__ float Ps[Br][Bc];      // per-tile attention weights p[ty][tx]

    // Running online-softmax state (per Q-row, in registers across tiles)
    float mi  = -1e9f;   // running maximum score
    float li  =  0.0f;   // running normalizer: Σ exp(s - mi)
    float Oi0 =  0.0f;   // running output accumulator for d-element tx*2
    float Oi1 =  0.0f;   // running output accumulator for d-element tx*2+1

    int tid_flat = ty * 32 + tx;   // flat thread index within block, 0..1023

    // ── Load Q tile into shared memory (once, before the KV loop) ────────────
    // Each thread loads 2 elements; all 1024 threads together cover Br*d = 2048.
    // Access: Qh[row * d + c], warp threads load consecutive c → coalesced.
    #pragma unroll 2
    for (int idx = tid_flat; idx < Br * d; idx += 1024) {
        int r       = idx / d;
        int c       = idx % d;
        int abs_row = q_tile * Br + r;
        Qs[r][c] = (abs_row < N) ? Qh[abs_row * d + c] : 0.0f;
    }
    __syncthreads();

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int kv_base = kv_tile * Bc;

        // ── Load K and V tiles cooperatively ─────────────────────────────────
        // Ks[r][c] writes only columns 0..63; column 64 (padding) left uninitialized.
        // V has no padding (accessed as Vs[j][tx*2], accepted minor 2-way conflict).
        #pragma unroll 2
        for (int idx = tid_flat; idx < Bc * d; idx += 1024) {
            int r      = idx / d;
            int c      = idx % d;
            int kv_row = kv_base + r;
            Ks[r][c] = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
            Vs[r][c] = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
        }
        __syncthreads();

        // ── Phase 1: Each thread computes one complete dot product ────────────
        // S[ty][tx] = Σ_{i=0}^{63} Q[ty][i] * K[tx][i]
        //
        // Qs[ty][i]: all 32 threads in warp (same ty) access same address → broadcast
        //            (1 shared-mem transaction per iteration, 0 conflicts)
        // Ks[tx][i]: thread tx accesses Ks[tx][i]. With +1 padding (stride=65):
        //            bank = (tx*65 + i) % 32 = (tx + i) % 32
        //            For tx=0..31, fixed i: banks {i, i+1, ..., i+31} mod 32 = all 32 banks
        //            → zero bank conflicts! ✓
        //
        // 64 FMAs per thread, fully pipelined — NO warp reduction for S.
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 64; i++)
            s += Qs[ty][i] * Ks[tx][i];
        s *= scale;

        // Mask padded KV positions so they don't corrupt the softmax
        if (kv_base + tx >= N) s = -1e9f;

        // ── Phase 2: Row-wise online softmax — only 2 warp reductions per tile!
        // Reduction 1: row maximum (for numerical stability)
        float row_max = s;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            row_max = fmaxf(row_max, __shfl_down_sync(0xffffffff, row_max, off));
        row_max = __shfl_sync(0xffffffff, row_max, 0);  // broadcast to all lanes

        float new_mi = fmaxf(mi, row_max);
        float alpha  = expf(mi  - new_mi);   // rescale factor for old accumulator
        float p      = expf(s   - new_mi);   // attention weight for this (ty, tx) pair

        // Reduction 2: row sum of p (for normalizer update)
        float row_sum = p;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            row_sum += __shfl_down_sync(0xffffffff, row_sum, off);
        row_sum = __shfl_sync(0xffffffff, row_sum, 0);

        Ps[ty][tx] = p;           // write weight to shared; Phase 3 reads this
        li  = li  * alpha + row_sum;
        mi  = new_mi;
        __syncthreads();          // Ps fully visible before Phase 3 reads it

        // ── Phase 3: V accumulation with thread remapping ────────────────────
        // tx now acts as d-dimension index: thread tx owns output elements [tx*2, tx*2+1].
        // Inner loop: for each KV column j, accumulate p[ty][j] * V[j][tx*2..+1].
        //
        // Ps[ty][j]: all 32 warp-threads (same ty) read same address → broadcast ✓
        // Vs[j][tx*2]: stride-2 access, 2-way bank conflict (accepted; not critical path)
        float o0 = 0.0f, o1 = 0.0f;
        #pragma unroll
        for (int j = 0; j < Bc; j++) {
            float pj = Ps[ty][j];          // broadcast read
            o0 += pj * Vs[j][tx * 2];
            o1 += pj * Vs[j][tx * 2 + 1];
        }
        Oi0 = Oi0 * alpha + o0;
        Oi1 = Oi1 * alpha + o1;
        __syncthreads();   // guard next tile's Ks/Vs load from overwriting before reads finish
    }

    // ── Write normalized output ───────────────────────────────────────────────
    int q_row = q_tile * Br + ty;
    if (q_row < N) {
        float inv_li = 1.0f / li;
        Oh[q_row * d + tx * 2]     = Oi0 * inv_li;
        Oh[q_row * d + tx * 2 + 1] = Oi1 * inv_li;
    }
}

extern "C" void flash_attention_v6(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    dim3 block(32, Br);
    flash_attention_v6_kernel<<<grid, block>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
