#include <cuda_runtime.h>
#include <math.h>

#define Br 32
#define Bc 32

// K5: Two-Phase Within-Tile FlashAttention (Register-Tiled Softmax)
//
// Root cause of K1-K4 low performance (~1% compute utilization):
//   The inner KV-loop has a SERIAL DEPENDENCY CHAIN per j-step:
//     s_j = dot(Qi, Kj) + warp_reduce        ← sync
//     new_mi = max(mi, s_j)                   ← depends on s_j
//     alpha  = exp(mi - new_mi)               ← ~30 cycles, depends on new_mi
//     p      = exp(s_j - new_mi)              ← ~30 cycles, depends on s_j
//     Oi += p * Vj; li = li*alpha + p; mi = new_mi  ← depends on both exps
//   32 steps × serial chain = 32 × ~70 cycles = ~2240 cycles of sequential work.
//   GPU FLOP throughput: ~2% utilized.
//
// K5 solution: Split each KV-tile into two decoupled phases.
//
//   Phase 1 — Score computation (NO exp):
//     for j in 0..Bc-1:
//       s[j] = warp_reduce(Qi·Kj) * scale   ← store in register array
//       m_tile = max(m_tile, s[j])            ← only max(), no exp
//
//   Phase 2 — Independent weight + accumulation (PARALLEL exp):
//     for j in 0..Bc-1:
//       p = exp(s[j] - m_tile)               ← ALL 32 exp calls INDEPENDENT
//       Ot += p * Vj; li_tile += p            ← can pipeline across j
//
//   Phase 3 — Single tile-boundary rescaling (only 2 exp calls total):
//     new_mi = max(mi, m_tile)
//     Oi = Oi * exp(mi - new_mi) + Ot * exp(m_tile - new_mi)
//     li = li * exp(mi - new_mi) + li_tile * exp(m_tile - new_mi)
//
// exp() count per tile:
//   K3/K4: 2 * Bc = 64  (serial: alpha+p each step)
//   K5:    Bc + 2 = 34  (parallel: 1 per step, 2 at boundary)
//   → ~2× fewer exp calls, and Phase 2's 32 exp calls are fully independent
//   → GPU can issue multiple exp operations simultaneously (ILP)
//
// Register array s[Bc=32] stays in RF via full loop unrolling (#pragma unroll).
// Shared memory: 16KB (same as K1-K3), allowing 2 blocks/SM in principle.
// Launch bounds: (1024, 1) → 1 block/SM, up to 64 regs/thread (fits our ~52).

__global__ __launch_bounds__(1024, 1)
void flash_attention_v5_kernel(
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

    int lane  = threadIdx.x;   // 0..31, covers d/32 = 2 elements of Q/K/O
    int row   = threadIdx.y;   // 0..Br-1, which Q-row within this tile
    int q_row = q_tile * Br + row;
    int base  = lane * 2;      // start index in d-dimension for this thread

    float scale = 1.0f / sqrtf((float)d);

    // Global running softmax state (accumulated across all KV-tiles)
    float mi  = -1e9f;   // running maximum of scores seen so far
    float li  =  0.0f;   // running normalizer: sum_j exp(s_j - mi)
    float Oi0 =  0.0f;   // running output accumulator, d-element [base]
    float Oi1 =  0.0f;   // running output accumulator, d-element [base+1]

    // Load this thread's Q slice into registers — reused across all Tc tiles
    float Qi0 = 0.0f, Qi1 = 0.0f;
    if (q_row < N) {
        if (base     < d) Qi0 = Qh[q_row * d + base];
        if (base + 1 < d) Qi1 = Qh[q_row * d + base + 1];
    }

    // Shared memory for one KV-tile (same footprint as K1-K3: 16KB)
    __shared__ float Ks[Bc][64];
    __shared__ float Vs[Bc][64];

    int tid_flat = row * 32 + lane;
    int total    = Bc * d;   // 32 * 64 = 2048 elements per tile

    for (int kv_tile = 0; kv_tile < Tc; kv_tile++) {
        int kv_base = kv_tile * Bc;

        // ── Load K/V tile cooperatively (all 1024 threads pitch in) ────────
        for (int idx = tid_flat; idx < total; idx += 1024) {
            int r      = idx / d;
            int c      = idx % d;
            int kv_row = kv_base + r;
            Ks[r][c] = (kv_row < N) ? Kh[kv_row * d + c] : 0.0f;
            Vs[r][c] = (kv_row < N) ? Vh[kv_row * d + c] : 0.0f;
        }
        __syncthreads();

        // ── Phase 1: Compute all Bc scores, find tile max (NO exp) ─────────
        // s[] is a register array. Full unroll ensures static indexing →
        // compiler keeps all 32 floats in RF (no local memory spill).
        float s[Bc];
        float m_tile = -1e9f;

        #pragma unroll
        for (int j = 0; j < Bc; j++) {
            // Each thread holds 2 elements of the d-length dot product
            float sj  = Qi0 * Ks[j][base] + Qi1 * Ks[j][base + 1];

            // Warp reduction: sum partial dot products across all 32 lanes
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                sj += __shfl_down_sync(0xffffffff, sj, off);
            sj = __shfl_sync(0xffffffff, sj, 0);  // broadcast lane-0 to all
            sj *= scale;

            s[j]   = sj;                   // stash in register (unrolled → RF)
            m_tile = fmaxf(m_tile, sj);    // track tile-local max (no exp!)
        }

        // ── Phase 2: Independent exp + accumulate per-tile output ──────────
        // All 32 exp(s[j] - m_tile) calls are INDEPENDENT of each other.
        // The GPU scheduler can issue multiple exp units simultaneously,
        // hiding the ~30-cycle latency of each exp via instruction pipelining.
        float Ot0    = 0.0f;
        float Ot1    = 0.0f;
        float li_tile = 0.0f;

        #pragma unroll
        for (int j = 0; j < Bc; j++) {
            float pj   = expf(s[j] - m_tile);   // independent across j!
            Ot0       += pj * Vs[j][base];
            Ot1       += pj * Vs[j][base + 1];
            li_tile   += pj;
        }

        // ── Phase 3: Single tile-boundary rescaling (only 2 exp calls) ─────
        float new_mi = fmaxf(mi, m_tile);
        float alpha  = expf(mi     - new_mi);   // shrink old accumulator
        float beta   = expf(m_tile - new_mi);   // scale new tile's contribution

        Oi0 = Oi0 * alpha + Ot0    * beta;
        Oi1 = Oi1 * alpha + Ot1    * beta;
        li  = li  * alpha + li_tile * beta;
        mi  = new_mi;

        __syncthreads();
    }

    // Normalize and write output
    if (q_row < N) {
        float inv_li = 1.0f / li;
        if (base     < d) Oh[q_row * d + base]     = Oi0 * inv_li;
        if (base + 1 < d) Oh[q_row * d + base + 1] = Oi1 * inv_li;
    }
}

extern "C" void flash_attention_v5(
    const float* Q, const float* K, const float* V,
    float* O,
    int bh, int N, int d)
{
    int Tr = (N + Br - 1) / Br;
    dim3 grid(Tr, bh);
    dim3 block(32, Br);
    flash_attention_v5_kernel<<<grid, block>>>(Q, K, V, O, bh, N, d);
    cudaDeviceSynchronize();
}
