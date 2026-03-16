#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Benchmark config
// B=1, NH=8 -> bh=8 blocks per Tr tile; P100 has 56 SMs
// For N=1024, Tr=32: 8*32=256 concurrent blocks -> good SM utilization
#define BATCH    1
#define NH       8       // number of attention heads
#define HEAD_DIM 64      // d, must be multiple of 4
#define WARMUP   5
#define REPEAT   20

static const int N_LIST[] = {512, 1024, 2048, 4096};
static const int N_COUNT  = 4;

// Parameter names omitted to avoid macro collision
extern "C" {
    void naive_attention(
        const float*, const float*, const float*,
        float*, float*, int, int, int);
    void flash_attention_v1(
        const float*, const float*, const float*,
        float*, int, int, int);
    void flash_attention_v2(
        const float*, const float*, const float*,
        float*, int, int, int);
    void flash_attention_v3(
        const float*, const float*, const float*,
        float*, int, int, int);
    void flash_attention_v4(
        const float*, const float*, const float*,
        float*, int, int, int);
}

static void rand_fill(float* buf, int n) {
    unsigned s = 20250315u;
    for (int i = 0; i < n; i++) {
        s = s * 1664525u + 1013904223u;
        buf[i] = (float)(s >> 8 & 0xFFFFFF) / (float)0xFFFFFF - 0.5f;
    }
}

static float max_abs_err(const float* a, const float* b, int n) {
    float e = 0.0f;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(a[i] - b[i]);
        if (diff > e) e = diff;
    }
    return e;
}

// Each run gets its own cudaEvent pair for accurate per-iteration timing
#define COLLECT_TIMES(out, call)                                    \
    do {                                                            \
        for (int _w = 0; _w < WARMUP; _w++) { call; }             \
        cudaDeviceSynchronize();                                    \
        for (int _r = 0; _r < REPEAT; _r++) {                     \
            cudaEvent_t _t0, _t1;                                  \
            cudaEventCreate(&_t0);                                  \
            cudaEventCreate(&_t1);                                  \
            cudaEventRecord(_t0);                                   \
            call;                                                   \
            cudaEventRecord(_t1);                                   \
            cudaEventSynchronize(_t1);                              \
            cudaEventElapsedTime(&(out)[_r], _t0, _t1);            \
            cudaEventDestroy(_t0);                                  \
            cudaEventDestroy(_t1);                                  \
        }                                                           \
    } while (0)

static void compute_stats(const float* t, float* mean, float* sd) {
    float sum = 0.0f;
    for (int i = 0; i < REPEAT; i++) sum += t[i];
    *mean = sum / REPEAT;
    float var = 0.0f;
    for (int i = 0; i < REPEAT; i++) var += (t[i] - *mean) * (t[i] - *mean);
    *sd = sqrtf(var / REPEAT);
}

// ── DRAM traffic accounting ────────────────────────────────────────────────────
//
// K0 — Naive Attention (three separate kernel launches, materializes N×N S matrix):
//   Pass 1 (compute_S): read Q(N*d) + read K(N*d) + write S(N²)  = 2Nd + N²
//   Pass 2 (softmax)  : read S(N²) + write S(N²)                 = 2N²
//   Pass 3 (compute_O): read S(N²) + read V(N*d) + write O(N*d)  = N² + 2Nd
//   Total/head = 4*N² + 4*N*d
//     ← dominated by the N×N attention matrix
//
// K1/K2/K3 — FlashAttention (NO N×N matrix ever touches HBM):
//   The S tile (Br×Bc) lives entirely in registers/SRAM.
//   Only Q, K, V, O are read/written to HBM, each exactly once:
//     read Q(N*d) + read K(N*d) + read V(N*d) + write O(N*d) = 4*N*d
//   This is the theoretical minimum — independent of Br/Bc/tiling.
//   N=4096, d=64, bh=8: 4×4096×64×8×4 = 32 MB  (vs K0's ~2 GB)

#define P100_BW_GBs 732.0f

static long long dram_k0_bytes(int bh, int N, int d) {
    return (long long)bh * (4LL*N*N + 4LL*N*d) * (long long)sizeof(float);
}
static long long dram_flash_bytes(int bh, int N, int d) {
    // Q + K + V + O, each read/written exactly once. No N×N matrix.
    return (long long)bh * 4LL * N * d * (long long)sizeof(float);
}
static float bw_from_bytes(float ms, long long bytes) {
    return (float)((double)bytes / (ms * 1e-3) / 1e9);
}

struct Result {
    const char* name;
    float mean_ms;
    float sd_ms;
    float bw;
    float dram_mb;   // theoretical DRAM traffic in MB
    float err;
};

static void bench_one_n(int bh, int N, int d, FILE* csv) {
    size_t nd  = (size_t)bh * N * d * sizeof(float);
    size_t nn  = (size_t)bh * N * N * sizeof(float);

    float *hQ     = (float*)malloc(nd);
    float *hK     = (float*)malloc(nd);
    float *hV     = (float*)malloc(nd);
    float *hO_ref = (float*)malloc(nd);
    float *hO     = (float*)malloc(nd);
    rand_fill(hQ, bh * N * d);
    rand_fill(hK, bh * N * d);
    rand_fill(hV, bh * N * d);

    float *dQ, *dK, *dV, *dO, *dS;
    cudaMalloc(&dQ, nd); cudaMalloc(&dK, nd); cudaMalloc(&dV, nd);
    cudaMalloc(&dO, nd);
    cudaMalloc(&dS, nn);

    cudaMemcpy(dQ, hQ, nd, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, nd, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, nd, cudaMemcpyHostToDevice);

    float t[REPEAT];
    struct Result res[5];

    // DRAM traffic (theoretical)
    long long bytes_k0    = dram_k0_bytes(bh, N, d);
    long long bytes_flash = dram_flash_bytes(bh, N, d);

    // K0: reference
    naive_attention(dQ, dK, dV, dS, dO, bh, N, d);
    cudaMemcpy(hO_ref, dO, nd, cudaMemcpyDeviceToHost);
    COLLECT_TIMES(t, naive_attention(dQ, dK, dV, dS, dO, bh, N, d));
    compute_stats(t, &res[0].mean_ms, &res[0].sd_ms);
    res[0].name    = "K0: Naive Attention ";
    res[0].bw      = bw_from_bytes(res[0].mean_ms, bytes_k0);
    res[0].dram_mb = (float)((double)bytes_k0 / 1e6);
    res[0].err     = 0.0f;

    COLLECT_TIMES(t, flash_attention_v1(dQ, dK, dV, dO, bh, N, d));
    cudaMemcpy(hO, dO, nd, cudaMemcpyDeviceToHost);
    compute_stats(t, &res[1].mean_ms, &res[1].sd_ms);
    res[1].name    = "K1: Basic FlashAttn ";
    res[1].bw      = bw_from_bytes(res[1].mean_ms, bytes_flash);
    res[1].dram_mb = (float)((double)bytes_flash / 1e6);
    res[1].err     = max_abs_err(hO, hO_ref, bh * N * d);

    COLLECT_TIMES(t, flash_attention_v2(dQ, dK, dV, dO, bh, N, d));
    cudaMemcpy(hO, dO, nd, cudaMemcpyDeviceToHost);
    compute_stats(t, &res[2].mean_ms, &res[2].sd_ms);
    res[2].name    = "K2: +Reg+float4     ";
    res[2].bw      = bw_from_bytes(res[2].mean_ms, bytes_flash);
    res[2].dram_mb = (float)((double)bytes_flash / 1e6);
    res[2].err     = max_abs_err(hO, hO_ref, bh * N * d);

    COLLECT_TIMES(t, flash_attention_v3(dQ, dK, dV, dO, bh, N, d));
    cudaMemcpy(hO, dO, nd, cudaMemcpyDeviceToHost);
    compute_stats(t, &res[3].mean_ms, &res[3].sd_ms);
    res[3].name    = "K3: +NoBkConf+CoopLd";
    res[3].bw      = bw_from_bytes(res[3].mean_ms, bytes_flash);
    res[3].dram_mb = (float)((double)bytes_flash / 1e6);
    res[3].err     = max_abs_err(hO, hO_ref, bh * N * d);

    COLLECT_TIMES(t, flash_attention_v4(dQ, dK, dV, dO, bh, N, d));
    cudaMemcpy(hO, dO, nd, cudaMemcpyDeviceToHost);
    compute_stats(t, &res[4].mean_ms, &res[4].sd_ms);
    res[4].name    = "K4: +WarpDSplit     ";
    res[4].bw      = bw_from_bytes(res[4].mean_ms, bytes_flash);
    res[4].dram_mb = (float)((double)bytes_flash / 1e6);
    res[4].err     = max_abs_err(hO, hO_ref, bh * N * d);

    // print
    float base = res[0].mean_ms;
    printf("N=%-5d  d=%-3d  bh=%d  Br: K1=64, K2/K3/K4=128\n", N, d, bh);
    printf("  %-22s  %9s %7s  %9s  %9s %6s  %8s  %10s  %5s\n",
           "Kernel", "Mean(ms)", "Std(ms)",
           "DRAM(MB)", "BW(GB/s)", "Util%", "Speedup", "MaxAbsErr", "Pass");
    printf("  %.104s\n",
           "--------------------------------------------------------------------------------------------------------");
    for (int i = 0; i < 5; i++) {
        const char* pass = (i == 0) ? " ref"
                         : (res[i].err < 1e-2f ? " YES" : "  NO");
        float util = res[i].bw / P100_BW_GBs * 100.0f;
        printf("  %-22s  %9.3f %7.3f  %9.1f  %9.1f %5.1f%%  %7.2fx  %10.2e  %5s\n",
               res[i].name,
               res[i].mean_ms, res[i].sd_ms,
               res[i].dram_mb,
               res[i].bw, util,
               base / res[i].mean_ms,
               res[i].err, pass);
    }
    printf("\n");

    for (int i = 0; i < 5; i++) {
        const char* pass = (i == 0) ? "ref"
                         : (res[i].err < 1e-2f ? "YES" : "NO");
        fprintf(csv, "%d,%d,%d,%d,%s,%.4f,%.4f,%.1f,%.2f,%.4f,%.2e,%s\n",
                BATCH, NH, N, d, res[i].name,
                res[i].mean_ms, res[i].sd_ms,
                res[i].dram_mb, res[i].bw,
                base / res[i].mean_ms,
                res[i].err, pass);
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dS);
    free(hQ); free(hK); free(hV); free(hO_ref); free(hO);
}

int main(void) {
    int bh = BATCH * NH;
    int d  = HEAD_DIM;

    printf("\n[FlashAttention Benchmark]  batch=%d  heads=%d  bh=%d  d=%d  "
           "warmup=%d  repeat=%d\n\n",
           BATCH, NH, bh, d, WARMUP, REPEAT);

    FILE* csv = fopen("results.csv", "w");
    if (csv)
        fprintf(csv, "Batch,NH,N,d,Kernel,Mean_ms,Std_ms,DRAM_MB,"
                     "DRAM_BW_GBs,Speedup,MaxAbsErr,Pass\n");

    for (int i = 0; i < N_COUNT; i++)
        bench_one_n(bh, N_LIST[i], d, csv);

    if (csv) {
        fclose(csv);
        printf("results saved to results.csv\n");
    } else {
        printf("warning: could not write results.csv\n");
    }
    return 0;
}
