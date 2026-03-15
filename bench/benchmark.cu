#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define HEAD_DIM 64     // d, must be multiple of 4 (float4 alignment)
#define WARMUP   5      // discarded runs to warm up GPU caches
#define REPEAT   20     // timed runs; report mean +/- stddev

// N values to sweep
static const int N_LIST[] = {512, 1024, 2048, 4096};
static const int N_COUNT  = 4;

extern "C" {
    void naive_attention(
        const float*, const float*, const float*,
        float*, float*, int, int);
    void flash_attention_v1(
        const float*, const float*, const float*,
        float*, float*, float*, int, int);
    void flash_attention_v2(
        const float*, const float*, const float*,
        float*, int, int);
    void flash_attention_v3(
        const float*, const float*, const float*,
        float*, int, int);
}

// ── helpers ────────────────────────────────────────────────────────────────────

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

// Collect REPEAT individual timings (ms) into out[].
// Each run is bracketed by its own cudaEvent pair for accuracy.
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

static void stats(const float* t, float* mean, float* sd) {
    float sum = 0.0f;
    for (int i = 0; i < REPEAT; i++) sum += t[i];
    *mean = sum / REPEAT;
    float var = 0.0f;
    for (int i = 0; i < REPEAT; i++) var += (t[i] - *mean) * (t[i] - *mean);
    *sd = sqrtf(var / REPEAT);
}

// Lower-bound DRAM traffic estimates (bytes):
//   Naive : S is written once and read twice across three kernels,
//           total ~(4*n^2 + 4*n*d) * sizeof(float)
//   Flash : ideal minimum = Q+K+V read + O write = 4*n*d*sizeof(float)
static float bw_naive(float ms, int n, int d) {
    double bytes = (4.0 * n * n + 4.0 * n * d) * sizeof(float);
    return (float)(bytes / (ms * 1e-3) / 1e9);
}
static float bw_flash(float ms, int n, int d) {
    double bytes = 4.0 * n * d * sizeof(float);
    return (float)(bytes / (ms * 1e-3) / 1e9);
}

// ── per-N benchmark ────────────────────────────────────────────────────────────

struct Result {
    const char* name;
    float mean_ms;
    float sd_ms;
    float bw;
    float err;
};

static void bench_one_n(int n, int d, FILE* csv) {
    size_t nd  = (size_t)n * d * sizeof(float);
    size_t nn  = (size_t)n * n * sizeof(float);

    // host buffers
    float *hQ     = (float*)malloc(nd);
    float *hK     = (float*)malloc(nd);
    float *hV     = (float*)malloc(nd);
    float *hO_ref = (float*)malloc(nd);
    float *hO     = (float*)malloc(nd);
    rand_fill(hQ, n * d);
    rand_fill(hK, n * d);
    rand_fill(hV, n * d);

    // device buffers
    float *dQ, *dK, *dV, *dO, *dS, *dl, *dm;
    cudaMalloc(&dQ, nd); cudaMalloc(&dK, nd); cudaMalloc(&dV, nd);
    cudaMalloc(&dO, nd);
    cudaMalloc(&dS, nn);   // K0 only
    cudaMalloc(&dl, n * sizeof(float));  // K1 only
    cudaMalloc(&dm, n * sizeof(float));  // K1 only

    cudaMemcpy(dQ, hQ, nd, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, nd, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, nd, cudaMemcpyHostToDevice);

    float t[REPEAT];
    struct Result res[4];

    // K0 — also captures reference output
    naive_attention(dQ, dK, dV, dS, dO, n, d);
    cudaMemcpy(hO_ref, dO, nd, cudaMemcpyDeviceToHost);
    COLLECT_TIMES(t, naive_attention(dQ, dK, dV, dS, dO, n, d));
    stats(t, &res[0].mean_ms, &res[0].sd_ms);
    res[0].name = "K0: Naive Attention ";
    res[0].bw   = bw_naive(res[0].mean_ms, n, d);
    res[0].err  = 0.0f;

    // K1
    COLLECT_TIMES(t, flash_attention_v1(dQ, dK, dV, dO, dl, dm, n, d));
    cudaMemcpy(hO, dO, nd, cudaMemcpyDeviceToHost);
    stats(t, &res[1].mean_ms, &res[1].sd_ms);
    res[1].name = "K1: Basic FlashAttn ";
    res[1].bw   = bw_flash(res[1].mean_ms, n, d);
    res[1].err  = max_abs_err(hO, hO_ref, n * d);

    // K2
    COLLECT_TIMES(t, flash_attention_v2(dQ, dK, dV, dO, n, d));
    cudaMemcpy(hO, dO, nd, cudaMemcpyDeviceToHost);
    stats(t, &res[2].mean_ms, &res[2].sd_ms);
    res[2].name = "K2: +Reg+float4     ";
    res[2].bw   = bw_flash(res[2].mean_ms, n, d);
    res[2].err  = max_abs_err(hO, hO_ref, n * d);

    // K3
    COLLECT_TIMES(t, flash_attention_v3(dQ, dK, dV, dO, n, d));
    cudaMemcpy(hO, dO, nd, cudaMemcpyDeviceToHost);
    stats(t, &res[3].mean_ms, &res[3].sd_ms);
    res[3].name = "K3: +No BankConflict";
    res[3].bw   = bw_flash(res[3].mean_ms, n, d);
    res[3].err  = max_abs_err(hO, hO_ref, n * d);

    // print
    float base = res[0].mean_ms;
    printf("N=%-5d  d=%-3d\n", n, d);
    printf("  %-22s  %9s %8s  %14s  %8s  %10s  %5s\n",
           "Kernel", "Mean(ms)", "Std(ms)", "DRAM BW(GB/s)", "Speedup", "MaxAbsErr", "Pass");
    printf("  %.80s\n",
           "--------------------------------------------------------------------------------");
    for (int i = 0; i < 4; i++) {
        const char* pass = (i == 0) ? " ref"
                         : (res[i].err < 1e-2f ? " YES" : "  NO");
        printf("  %-22s  %9.3f %8.3f  %14.1f  %7.2fx  %10.2e  %5s\n",
               res[i].name,
               res[i].mean_ms, res[i].sd_ms,
               res[i].bw,
               base / res[i].mean_ms,
               res[i].err, pass);
    }
    printf("\n");

    // csv rows
    for (int i = 0; i < 4; i++) {
        const char* pass = (i == 0) ? "ref"
                         : (res[i].err < 1e-2f ? "YES" : "NO");
        fprintf(csv, "%d,%d,%s,%.4f,%.4f,%.2f,%.4f,%.2e,%s\n",
                n, d, res[i].name,
                res[i].mean_ms, res[i].sd_ms,
                res[i].bw,
                base / res[i].mean_ms,
                res[i].err, pass);
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dS); cudaFree(dl); cudaFree(dm);
    free(hQ); free(hK); free(hV); free(hO_ref); free(hO);
}

// ── main ───────────────────────────────────────────────────────────────────────

int main(void) {
    int d = HEAD_DIM;

    printf("\n[FlashAttention Benchmark]  d=%d  warmup=%d  repeat=%d\n\n",
           d, WARMUP, REPEAT);

    FILE* csv = fopen("results.csv", "w");
    if (csv)
        fprintf(csv, "N,d,Kernel,Mean_ms,Std_ms,DRAM_BW_GBs,Speedup,MaxAbsErr,Pass\n");

    for (int i = 0; i < N_COUNT; i++)
        bench_one_n(N_LIST[i], d, csv);

    if (csv) {
        fclose(csv);
        printf("results saved to results.csv\n");
    } else {
        printf("warning: could not write results.csv\n");
    }
    return 0;
}
