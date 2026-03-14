"""
benchmark.py — 统一测试 K0~K3 的性能和正确性

使用方式：
    python bench/benchmark.py

依赖：ctypes（标准库），numpy，cuda-python 或直接用 subprocess 调
实际上：用 ctypes 加载各 kernel 的 .so，用 cudaMalloc/cudaMemcpy via ctypes
但更简单的方式是用 subprocess 调 nvcc 编译的可执行文件，
这里选择 ctypes + numpy，通过系统的 libcuda.so 和 libcudart.so 完成内存管理。

前提：已运行 make 编译生成 build/*.so
"""

import ctypes
import numpy as np
import time
import os
import sys
import subprocess

# ─── 路径 ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(ROOT, "build")

# ─── 加载 CUDA Runtime ─────────────────────────────────────────────────────────
cudart = ctypes.CDLL("libcudart.so", mode=ctypes.RTLD_GLOBAL)

cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int]

cudart.cudaFree.restype = ctypes.c_int
cudart.cudaFree.argtypes = [ctypes.c_void_p]

cudart.cudaDeviceSynchronize.restype = ctypes.c_int

# cudaMemcpyKind
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

def cuda_malloc(nbytes):
    ptr = ctypes.c_void_p()
    ret = cudart.cudaMalloc(ctypes.byref(ptr), nbytes)
    assert ret == 0, f"cudaMalloc failed: {ret}"
    return ptr

def cuda_free(ptr):
    cudart.cudaFree(ptr)

def to_device(arr):
    """把 numpy 数组拷到 GPU，返回 device pointer"""
    nbytes = arr.nbytes
    ptr = cuda_malloc(nbytes)
    ret = cudart.cudaMemcpy(ptr, arr.ctypes.data_as(ctypes.c_void_p),
                            nbytes, cudaMemcpyHostToDevice)
    assert ret == 0
    return ptr

def from_device(ptr, shape, dtype=np.float32):
    """从 GPU 拷回 numpy 数组"""
    arr = np.empty(shape, dtype=dtype)
    nbytes = arr.nbytes
    ret = cudart.cudaMemcpy(arr.ctypes.data_as(ctypes.c_void_p), ptr,
                            nbytes, cudaMemcpyDeviceToHost)
    assert ret == 0
    return arr

# ─── 参考实现（CPU numpy）────────────────────────────────────────────────────
def ref_attention(Q, K, V):
    d = Q.shape[-1]
    scale = 1.0 / np.sqrt(d)
    S = Q @ K.T * scale          # [N, N]
    S = S - S.max(axis=-1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=-1, keepdims=True)
    return (P @ V).astype(np.float32)

# ─── 计时工具 ──────────────────────────────────────────────────────────────────
def measure_time(fn, warmup=3, repeat=10):
    """GPU 同步计时，单位 ms"""
    for _ in range(warmup):
        fn()
    cudart.cudaDeviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    cudart.cudaDeviceSynchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / repeat * 1000  # ms

# ─── DRAM 带宽估算 ─────────────────────────────────────────────────────────────
def dram_bw(N, d, time_ms):
    """
    Naive: 读 Q/K/V + 读写 S(N²) + 写 O
    Flash: 读 Q(Tr次) + 读 K/V(Tr*Tc次) + 写 O
    这里用实际 IO 下界估算（bytes / time）
    """
    bytes_qkv = 3 * N * d * 4   # Q K V
    bytes_o   = N * d * 4       # O
    total_bytes = bytes_qkv + bytes_o
    return total_bytes / (time_ms * 1e-3) / 1e9  # GB/s

# ─── 各 Kernel 的 benchmark 函数 ───────────────────────────────────────────────

def bench_k0(N, d, d_Q, d_K, d_V):
    lib = ctypes.CDLL(os.path.join(BUILD_DIR, "naive_attention.so"))
    lib.naive_attention.restype = None
    lib.naive_attention.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # Q K V
        ctypes.c_void_p, ctypes.c_void_p,                    # S O
        ctypes.c_int, ctypes.c_int]

    d_S = cuda_malloc(N * N * 4)
    d_O = cuda_malloc(N * d * 4)

    def run():
        lib.naive_attention(d_Q, d_K, d_V, d_S, d_O, N, d)

    t = measure_time(run)
    O = from_device(d_O, (N, d))
    cuda_free(d_S); cuda_free(d_O)
    return t, O

def bench_k1(N, d, d_Q, d_K, d_V):
    lib = ctypes.CDLL(os.path.join(BUILD_DIR, "flash_attention_v1.so"))
    lib.flash_attention_v1.restype = None
    lib.flash_attention_v1.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # Q K V
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # O l m
        ctypes.c_int, ctypes.c_int]

    d_O = cuda_malloc(N * d * 4)
    d_l = cuda_malloc(N * 4)
    d_m = cuda_malloc(N * 4)

    def run():
        lib.flash_attention_v1(d_Q, d_K, d_V, d_O, d_l, d_m, N, d)

    t = measure_time(run)
    O = from_device(d_O, (N, d))
    cuda_free(d_O); cuda_free(d_l); cuda_free(d_m)
    return t, O

def bench_k2(N, d, d_Q, d_K, d_V):
    lib = ctypes.CDLL(os.path.join(BUILD_DIR, "flash_attention_v2.so"))
    lib.flash_attention_v2.restype = None
    lib.flash_attention_v2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # Q K V
        ctypes.c_void_p,                                      # O
        ctypes.c_int, ctypes.c_int]

    d_O = cuda_malloc(N * d * 4)

    def run():
        lib.flash_attention_v2(d_Q, d_K, d_V, d_O, N, d)

    t = measure_time(run)
    O = from_device(d_O, (N, d))
    cuda_free(d_O)
    return t, O

def bench_k3(N, d, d_Q, d_K, d_V):
    lib = ctypes.CDLL(os.path.join(BUILD_DIR, "flash_attention_v3.so"))
    lib.flash_attention_v3.restype = None
    lib.flash_attention_v3.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,  # Q K V
        ctypes.c_void_p,                                      # O
        ctypes.c_int, ctypes.c_int]

    d_O = cuda_malloc(N * d * 4)

    def run():
        lib.flash_attention_v3(d_Q, d_K, d_V, d_O, N, d)

    t = measure_time(run)
    O = from_device(d_O, (N, d))
    cuda_free(d_O)
    return t, O

# ─── 主程序 ────────────────────────────────────────────────────────────────────

def main():
    N = 1024    # 序列长度
    d = 64      # head dimension，必须是 4 的倍数（float4 要求）

    print(f"配置：N={N}, d={d}, dtype=float32\n")

    # 生成随机输入
    np.random.seed(42)
    Q = np.random.randn(N, d).astype(np.float32)
    K = np.random.randn(N, d).astype(np.float32)
    V = np.random.randn(N, d).astype(np.float32)

    # 上传到 GPU
    d_Q = to_device(Q)
    d_K = to_device(K)
    d_V = to_device(V)

    # CPU 参考值
    O_ref = ref_attention(Q, K, V)

    kernels = [
        ("K0: Naive Attention   ", bench_k0),
        ("K1: Basic FlashAttn   ", bench_k1),
        ("K2: +Reg+float4       ", bench_k2),
        ("K3: +No BankConflict  ", bench_k3),
    ]

    results = []
    for name, fn in kernels:
        try:
            t, O = fn(N, d, d_Q, d_K, d_V)
            # 正确性验证（相对误差 < 1e-3 即通过）
            ok = np.allclose(O, O_ref, atol=1e-2, rtol=1e-3)
            bw = dram_bw(N, d, t)
            results.append((name, t, bw, ok))
        except Exception as e:
            results.append((name, None, None, str(e)))

    cuda_free(d_Q); cuda_free(d_K); cuda_free(d_V)

    # 输出表格
    base_t = results[0][1] if results[0][1] is not None else 1.0
    header = f"{'Kernel':<26} {'Time(ms)':>10} {'DRAM BW(GB/s)':>15} {'Speedup':>10} {'Correct':>8}"
    print(header)
    print("-" * len(header))
    for name, t, bw, ok in results:
        if t is None:
            print(f"{name:<26} {'ERROR':>10}  {str(ok)}")
        else:
            speedup = base_t / t
            correct = "✓" if ok is True else ("✗" if ok is False else "?")
            print(f"{name:<26} {t:>10.2f} {bw:>15.1f} {speedup:>9.2f}x {correct:>8}")

if __name__ == "__main__":
    main()
