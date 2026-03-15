#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// d must be a multiple of 4 (float4 alignment)
#define SEQ_LEN 1024
#define HEAD_DIM 64
#define WARMUP   3
#define REPEAT   10

// Parameter names omitted to avoid collision with macros
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

// LCG random fill, uniform in [-0.5, 0.5]
static void rand_fill(float* buf, int n) {
    unsigned s = 20250101u;
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

// Warmup WARMUP times, then time REPEAT runs with cudaEvent
#define TIME_KERNEL(ms_var, call)                               \
    do {                                                        \
        for (int _i = 0; _i < WARMUP; _i++) { call; }         \
        cudaDeviceSynchronize();                                \
        cudaEvent_t _t0, _t1;                                  \
        cudaEventCreate(&_t0);                                  \
        cudaEventCreate(&_t1);                                  \
        cudaEventRecord(_t0);                                   \
        for (int _i = 0; _i < REPEAT; _i++) { call; }         \
        cudaEventRecord(_t1);                                   \
        cudaEventSynchronize(_t1);                              \
        cudaEventElapsedTime(&(ms_var), _t0, _t1);             \
        (ms_var) /= REPEAT;                                     \
        cudaEventDestroy(_t0);                                  \
        cudaEventDestroy(_t1);                                  \
    } while (0)

// Naive: Q/K/V read + S written + S read twice (softmax) + S read + V read + O write
//   ~= (4*n*n + 4*n*d) * sizeof(float)
// Flash: lower bound = Q+K+V read + O write = 4*n*d*sizeof(float)
static float bw_naive_GBs(float ms, int n, int d) {
    long bytes = (4LL * n * n + 4LL * n * d) * sizeof(float);
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}
static float bw_flash_GBs(float ms, int n, int d) {
    long bytes = 4LL * n * d * sizeof(float);
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}

int main(void) {
    int n = SEQ_LEN, d = HEAD_DIM;

    size_t nd_bytes = (size_t)n * d * sizeof(float);
    float *hQ     = (float*)malloc(nd_bytes);
    float *hK     = (float*)malloc(nd_bytes);
    float *hV     = (float*)malloc(nd_bytes);
    float *hO_ref = (float*)malloc(nd_bytes);
    float *hO     = (float*)malloc(nd_bytes);

    rand_fill(hQ, n * d);
    rand_fill(hK, n * d);
    rand_fill(hV, n * d);

    float *dQ, *dK, *dV, *dO, *dS, *dl, *dm;
    cudaMalloc(&dQ, nd_bytes);
    cudaMalloc(&dK, nd_bytes);
    cudaMalloc(&dV, nd_bytes);
    cudaMalloc(&dO, nd_bytes);
    cudaMalloc(&dS, (size_t)n * n * sizeof(float));  // K0 only
    cudaMalloc(&dl, n * sizeof(float));               // K1 only
    cudaMalloc(&dm, n * sizeof(float));               // K1 only

    cudaMemcpy(dQ, hQ, nd_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, nd_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, nd_bytes, cudaMemcpyHostToDevice);

    struct KernelResult { const char* name; float ms; float bw; float err; };
    struct KernelResult res[4];

    // K0: run once to capture reference output, then time
    {
        float ms;
        naive_attention(dQ, dK, dV, dS, dO, n, d);
        cudaMemcpy(hO_ref, dO, nd_bytes, cudaMemcpyDeviceToHost);
        TIME_KERNEL(ms, naive_attention(dQ, dK, dV, dS, dO, n, d));
        res[0].name = "K0: Naive Attention ";
        res[0].ms   = ms;
        res[0].bw   = bw_naive_GBs(ms, n, d);
        res[0].err  = 0.0f;
    }

    {
        float ms;
        TIME_KERNEL(ms, flash_attention_v1(dQ, dK, dV, dO, dl, dm, n, d));
        cudaMemcpy(hO, dO, nd_bytes, cudaMemcpyDeviceToHost);
        res[1].name = "K1: Basic FlashAttn ";
        res[1].ms   = ms;
        res[1].bw   = bw_flash_GBs(ms, n, d);
        res[1].err  = max_abs_err(hO, hO_ref, n * d);
    }

    {
        float ms;
        TIME_KERNEL(ms, flash_attention_v2(dQ, dK, dV, dO, n, d));
        cudaMemcpy(hO, dO, nd_bytes, cudaMemcpyDeviceToHost);
        res[2].name = "K2: +Reg+float4     ";
        res[2].ms   = ms;
        res[2].bw   = bw_flash_GBs(ms, n, d);
        res[2].err  = max_abs_err(hO, hO_ref, n * d);
    }

    {
        float ms;
        TIME_KERNEL(ms, flash_attention_v3(dQ, dK, dV, dO, n, d));
        cudaMemcpy(hO, dO, nd_bytes, cudaMemcpyDeviceToHost);
        res[3].name = "K3: +No BankConflict";
        res[3].ms   = ms;
        res[3].bw   = bw_flash_GBs(ms, n, d);
        res[3].err  = max_abs_err(hO, hO_ref, n * d);
    }

    float base = res[0].ms;
    printf("\n[FlashAttention Benchmark]  N=%d  d=%d  warmup=%d  repeat=%d\n\n",
           n, d, WARMUP, REPEAT);
    printf("%-22s  %9s  %14s  %8s  %10s  %5s\n",
           "Kernel", "Time(ms)", "DRAM BW(GB/s)", "Speedup", "MaxAbsErr", "Pass");
    printf("%.82s\n", "------------------------------------------------------------"
                      "------------------------");
    for (int i = 0; i < 4; i++) {
        const char* pass = (i == 0) ? " ref" : (res[i].err < 1e-2f ? " YES" : "  NO");
        printf("%-22s  %9.3f  %14.1f  %7.2fx  %10.2e  %5s\n",
               res[i].name, res[i].ms, res[i].bw,
               base / res[i].ms, res[i].err, pass);
    }
    printf("\n");

    FILE* fp = fopen("results.csv", "w");
    if (fp) {
        fprintf(fp, "Kernel,Time_ms,DRAM_BW_GBs,Speedup,MaxAbsErr,Pass\n");
        for (int i = 0; i < 4; i++) {
            const char* pass = (i == 0) ? "ref" : (res[i].err < 1e-2f ? "YES" : "NO");
            fprintf(fp, "%s,%.4f,%.2f,%.4f,%.2e,%s\n",
                    res[i].name, res[i].ms, res[i].bw,
                    base / res[i].ms, res[i].err, pass);
        }
        fclose(fp);
        printf("results saved to results.csv\n");
    } else {
        printf("warning: could not write results.csv\n");
    }

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dS); cudaFree(dl); cudaFree(dm);
    free(hQ); free(hK); free(hV); free(hO_ref); free(hO);
    return 0;
}
