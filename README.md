# FlashAttention CUDA — P100 优化实验

从零实现 FlashAttention，在 NVIDIA Tesla P100（Pascal, sm_60）上逐步优化，
展示从 Memory Bound 的 Naive Attention 到消除 Bank Conflict 的完整优化路径。
不依赖 PyTorch/cuBLAS，纯 CUDA C + ctypes 调用。

---

## 优化路径

| Kernel | 核心改动 | 瓶颈 |
|--------|----------|------|
| K0: Naive Attention | 显式分配 N×N 的 S 矩阵于 HBM，三趟 kernel | DRAM BW（S 矩阵 O(N²) 读写） |
| K1: Basic FlashAttention | Tiling + Online Softmax，消除 N×N HBM 写入；l/m 仍在 HBM | 多余的 l/m HBM 往返 |
| K2: +Reg + float4 | l/m 移入寄存器；Q/K/V 用 float4 向量化加载 | Shared Memory Bank Conflict |
| K3: +No BankConflict | Ss 矩阵每行 padding 1 个 float，消除列访问冲突 | — |

## 性能结果（N=1024, d=64, P100）

```
Kernel                     Time(ms)   DRAM BW(GB/s)    Speedup   Correct
K0: Naive Attention           xx.x            xx.x       1.00x       ✓
K1: Basic FlashAttn           xx.x            xx.x       x.xxx       ✓
K2: +Reg+float4               xx.x            xx.x       x.xxx       ✓
K3: +No BankConflict          xx.x            xx.x       x.xxx       ✓
```

*（运行 `python bench/benchmark.py` 填充实际数据）*

## Nsight Compute 关键指标

<!-- 截图占位：ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum ./build/xxx -->

| Metric | K0 | K1 | K2 | K3 |
|--------|----|----|----|----|
| DRAM Read (GB/s) | — | — | — | — |
| DRAM Write (GB/s) | — | — | — | — |
| Shared Mem Bank Conflict | — | — | — | — |
| Occupancy (%) | — | — | — | — |

---

## 文件结构

```
src/
  naive_attention.cu      # K0: 三趟 kernel，N×N S 矩阵写 HBM
  flash_attention_v1.cu   # K1: Tiling + Online Softmax，l/m 在 HBM
  flash_attention_v2.cu   # K2: l/m 进寄存器 + float4 向量化
  flash_attention_v3.cu   # K3: Ss padding 消除 Bank Conflict
bench/
  benchmark.py            # ctypes 调用，输出性能表格 + 正确性验证
Makefile
```

## 编译和运行

```bash
make          # 编译所有 kernel 到 build/*.so
# 或单独编译：
make k0 k1 k2 k3

python bench/benchmark.py
```

要求：CUDA 10.2+，sm_60，d 为 4 的倍数（float4 对齐）。

## Online Softmax 原理

标准 softmax 需要两遍扫描（先找 max，再算 exp sum）。
Online Softmax 合并为一遍，维护运行最大值 $m$ 和归一化因子 $l$：

$$
m_{\text{new}} = \max(m_{\text{old}}, m_{\text{block}})
$$
$$
l_{\text{new}} = l_{\text{old}} \cdot e^{m_{\text{old}} - m_{\text{new}}} + \sum_j e^{s_j - m_{\text{new}}}
$$

FlashAttention 利用这个性质，在 tile 间递推，无需实例化完整 S 矩阵。

## 参考

- [FlashAttention 论文](https://arxiv.org/abs/2205.14135) — Dao et al., 2022
- [flash-attention-minimal](https://github.com/tspeterkim/flash-attention-minimal) — K1 实现参考
