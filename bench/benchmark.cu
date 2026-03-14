/*
 * benchmark.cu — 统一测试 K0~K3 的性能和正确性
 *
 * 编译：见 Makefile 的 bench 目标，等价于：
 *   nvcc -O2 -arch=sm_60 bench/benchmark.cu src/naive_attention.cu \
 *        src/flash_attention_v1.cu src/flash_attention_v2.cu \
 *        src/flash_attention_v3.cu -o build/benchmark
 *
 * 运行：./build/benchmark
 * 输出：终端打印性能表格，同时写入 results.csv
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

// ─── 测试配置（d 必须是 4 的倍数，满足 float4 对齐）─────────────────────────────
#define N      1024
#define D      64
#define WARMUP 3
#define REPEAT 10

// ─── 各 kernel 的 C 接口声明（定义在各自的 .cu 文件中）─────────────────────────
extern "C" {
    void naive_attention(
        const float* Q, const float* K, const float* V,
        float* S, float* O, int N, int d);

    void flash_attention_v1(
        const float* Q, const float* K, const float* V,
        float* O, float* l, float* m, int N, int d);

    void flash_attention_v2(
        const float* Q, const float* K, const float* V,
        float* O, int N, int d);

    void flash_attention_v3(
        const float* Q, const float* K, const float* V,
        float* O, int N, int d);
}

// ─── 工具 ──────────────────────────────────────────────────────────────────────

// 用简单 LCG 初始化，生成 [-0.5, 0.5] 均匀分布
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
        float d = fabsf(a[i] - b[i]);
        if (d > e) e = d;
    }
    return e;
}

// cudaEvent 计时宏：先 WARMUP 次热身，再 REPEAT 次取平均
// 避免首次启动的 JIT / cache cold 干扰
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

// DRAM 访问量估算（下界，用于计算理论带宽利用率）
// Naive：Q/K/V 读 + S 写 + S 读两次（softmax in-place）+ S 读 + V 读 + O 写
//   = (3N×d + 4N×N + N×d) × 4 bytes，近似用 (4N²+4Nd)×4
// Flash：Q/K/V 读 + O 写 = 4N×d×4（不计 K/V 被 Tr 次重读，报理想下界）
static float bw_naive_GBs(float ms) {
    long bytes = (4LL * N * N + 4LL * N * D) * sizeof(float);
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}
static float bw_flash_GBs(float ms) {
    long bytes = 4LL * N * D * sizeof(float);
    return (float)bytes / (ms * 1e-3f) / 1e9f;
}

// ─── 主程序 ────────────────────────────────────────────────────────────────────
int main(void) {
    int n = N, d = D;

    // 1. 主机内存
    size_t nd_bytes = (size_t)n * d * sizeof(float);
    float *hQ = (float*)malloc(nd_bytes);
    float *hK = (float*)malloc(nd_bytes);
    float *hV = (float*)malloc(nd_bytes);
    float *hO_ref = (float*)malloc(nd_bytes);
    float *hO     = (float*)malloc(nd_bytes);

    rand_fill(hQ, n * d);
    rand_fill(hK, n * d);
    rand_fill(hV, n * d);

    // 2. 设备内存
    float *dQ, *dK, *dV, *dO, *dS, *dl, *dm;
    cudaMalloc(&dQ, nd_bytes);
    cudaMalloc(&dK, nd_bytes);
    cudaMalloc(&dV, nd_bytes);
    cudaMalloc(&dO, nd_bytes);
    cudaMalloc(&dS, (size_t)n * n * sizeof(float));  // 仅 K0 使用
    cudaMalloc(&dl, n * sizeof(float));               // 仅 K1 使用
    cudaMalloc(&dm, n * sizeof(float));               // 仅 K1 使用

    cudaMemcpy(dQ, hQ, nd_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, nd_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, nd_bytes, cudaMemcpyHostToDevice);

    // 3. 计时并记录结果
    struct { const char* name; float ms; float bw; float err; } res[4];

    // K0：先运行一次取参考值，再正式计时
    {
        float ms;
        naive_attention(dQ, dK, dV, dS, dO, n, d);  // 用于取参考
        cudaMemcpy(hO_ref, dO, nd_bytes, cudaMemcpyDeviceToHost);
        TIME_KERNEL(ms, naive_attention(dQ, dK, dV, dS, dO, n, d));
        res[0] = {"K0: Naive Attention ", ms, bw_naive_GBs(ms), 0.0f};
    }

    {
        float ms;
        TIME_KERNEL(ms, flash_attention_v1(dQ, dK, dV, dO, dl, dm, n, d));
        cudaMemcpy(hO, dO, nd_bytes, cudaMemcpyDeviceToHost);
        res[1] = {"K1: Basic FlashAttn ", ms, bw_flash_GBs(ms), max_abs_err(hO, hO_ref, n * d)};
    }

    {
        float ms;
        TIME_KERNEL(ms, flash_attention_v2(dQ, dK, dV, dO, n, d));
        cudaMemcpy(hO, dO, nd_bytes, cudaMemcpyDeviceToHost);
        res[2] = {"K2: +Reg+float4     ", ms, bw_flash_GBs(ms), max_abs_err(hO, hO_ref, n * d)};
    }

    {
        float ms;
        TIME_KERNEL(ms, flash_attention_v3(dQ, dK, dV, dO, n, d));
        cudaMemcpy(hO, dO, nd_bytes, cudaMemcpyDeviceToHost);
        res[3] = {"K3: +No BankConflict", ms, bw_flash_GBs(ms), max_abs_err(hO, hO_ref, n * d)};
    }

    // 4. 打印终端表格
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

    // 5. 写 CSV
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
        printf("结果已写入 results.csv\n");
    } else {
        printf("警告：无法写入 results.csv\n");
    }

    // 6. 释放
    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dS); cudaFree(dl); cudaFree(dm);
    free(hQ); free(hK); free(hV); free(hO_ref); free(hO);
    return 0;
}
