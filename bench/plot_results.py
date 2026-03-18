#!/usr/bin/env python3
"""
Generate benchmark charts from results.csv produced by ./build/benchmark.
Saves docs/benchmark_results.png (created automatically if it doesn't exist).

Usage:
    python bench/plot_results.py                  # reads results.csv in cwd
    python bench/plot_results.py path/to/results.csv
"""

import csv
import sys
import os
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────

CSV_PATH = sys.argv[1] if len(sys.argv) > 1 else "results.csv"
OUT_DIR  = "docs"
OUT_PATH = os.path.join(OUT_DIR, "benchmark_results.png")

# Canonical display order / colours
KERNEL_STYLE = {
    "K0: Naive Attention ": dict(color="#555555", ls="--",  marker="s", label="K0: Naive"),
    "K1: Basic FlashAttn ": dict(color="#1f77b4", ls="-",   marker="o", label="K1: Basic FA"),
    "K2: +Reg+float4     ": dict(color="#ff7f0e", ls="-",   marker="^", label="K2: +float4"),
    "K3: +NoBkConf+CoopLd": dict(color="#2ca02c", ls="-",   marker="D", label="K3: +NoBankConf"),
    "K4: +WarpDSplit     ": dict(color="#d62728", ls="-",   marker="v", label="K4: +DblBuf"),
    "K5: +2PhaseExpILP   ": dict(color="#9467bd", ls="-",   marker="P", label="K5: +2PhaseILP"),
    "K6: +GemmStyle      ": dict(color="#e377c2", ls="-",   marker="*", label="K6: +GemmStyle"),
}

# ── Load CSV ─────────────────────────────────────────────────────────────────

def load(path):
    data = collections.defaultdict(dict)   # data[kernel_name][N] = row_dict
    ns   = set()
    kernels = []
    with open(path) as f:
        for row in csv.DictReader(f):
            n  = int(row["N"])
            kn = row["Kernel"]
            data[kn][n] = row
            ns.add(n)
            if kn not in kernels:
                kernels.append(kn)
    return data, sorted(ns), kernels

data, NS, KERNELS = load(CSV_PATH)
NS_arr = np.array(NS)

os.makedirs(OUT_DIR, exist_ok=True)

# ── Build figure ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle(
    "FlashAttention CUDA — P100 Benchmark  (d=64, batch=1, heads=8)",
    fontsize=14, fontweight="bold", y=0.98
)

def style(kn):
    return KERNEL_STYLE.get(kn, dict(color="gray", ls="-", marker="x",
                                      label=kn.strip()))

# ── Panel 1: Speedup vs N ────────────────────────────────────────────────────
ax = axes[0, 0]
for kn in KERNELS:
    s = style(kn)
    ys = []
    for n in NS:
        row = data[kn].get(n)
        ys.append(float(row["Speedup"]) if row else None)
    ax.plot(NS, ys, color=s["color"], ls=s["ls"], marker=s["marker"],
            linewidth=1.8, markersize=7, label=s["label"])
ax.axhline(1.0, color="gray", lw=0.8, ls=":")
ax.set_xscale("log", base=2)
ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
ax.set_xlabel("Sequence length N"); ax.set_ylabel("Speedup vs K0")
ax.set_title("Speedup over Naive Attention")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3)

# ── Panel 2: Execution time vs N ─────────────────────────────────────────────
ax = axes[0, 1]
for kn in KERNELS:
    s = style(kn)
    ys = []
    for n in NS:
        row = data[kn].get(n)
        ys.append(float(row["Mean_ms"]) if row else None)
    ax.plot(NS, ys, color=s["color"], ls=s["ls"], marker=s["marker"],
            linewidth=1.8, markersize=7, label=s["label"])
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
ax.set_xlabel("Sequence length N"); ax.set_ylabel("Time (ms, log scale)")
ax.set_title("Execution Time")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, alpha=0.3, which="both")

# ── Panel 3: DRAM traffic (bar chart) ────────────────────────────────────────
ax = axes[1, 0]
# Show one representative N=1024 and N=4096 side-by-side per kernel
show_ns = [1024, 4096]
n_k = len(KERNELS)
x = np.arange(n_k)
bw = 0.35
offsets = [-bw/2, bw/2]
colors_bar = ["#4c72b0", "#dd8452"]
for idx, n in enumerate(show_ns):
    vals = [float(data[kn][n]["DRAM_MB"]) if n in data[kn] else 0 for kn in KERNELS]
    ax.bar(x + offsets[idx], vals, bw, label=f"N={n}",
           color=colors_bar[idx], alpha=0.8, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([s["label"] for s in [style(kn) for kn in KERNELS]],
                   rotation=25, ha="right", fontsize=8)
ax.set_ylabel("Theoretical DRAM Traffic (MB)")
ax.set_title("DRAM Traffic Reduction")
ax.set_yscale("log")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# ── Panel 4: Bandwidth utilisation % ─────────────────────────────────────────
ax = axes[1, 1]
for kn in KERNELS:
    s = style(kn)
    ys = []
    for n in NS:
        row = data[kn].get(n)
        if row:
            bw_val = float(row["DRAM_BW_GBs"])
            ys.append(bw_val / 732.0 * 100.0)
        else:
            ys.append(None)
    ax.plot(NS, ys, color=s["color"], ls=s["ls"], marker=s["marker"],
            linewidth=1.8, markersize=7, label=s["label"])
ax.set_xscale("log", base=2)
ax.set_xticks(NS); ax.set_xticklabels([str(n) for n in NS])
ax.set_xlabel("Sequence length N"); ax.set_ylabel("HBM Bandwidth Utilisation (%)")
ax.set_title("HBM Utilisation  (P100 peak = 732 GB/s)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
