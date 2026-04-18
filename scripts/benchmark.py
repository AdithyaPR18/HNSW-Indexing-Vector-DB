"""
Benchmark: HNSW vs brute-force cosine similarity

Metrics measured at each dataset size:
  - Build time (seconds)
  - Query throughput (queries/sec) for HNSW and brute-force
  - Recall@10 of HNSW vs ground-truth brute-force

Output: benchmark_results.png saved next to this script.
"""

from __future__ import annotations

import sys
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hnsw_index


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATASET_SIZES  = [1_000, 5_000, 10_000, 50_000, 100_000]
DIM            = 128
K              = 10
# ef scales with dataset — wider beam = better recall, lower throughput
def ef_for_size(n: int) -> int:
    if n <= 1_000:   return 64
    if n <= 5_000:   return 100
    if n <= 10_000:  return 150
    if n <= 50_000:  return 250
    return 350

EF_SEARCH      = -1   # placeholder; overridden per-size below
M              = 16
EF_CONST       = 200
N_QUERY        = 200   # queries per size for throughput / recall measurement
SEED           = 42

# Uncomment to go up to 1M (takes ~10–30 min on a laptop):
# DATASET_SIZES += [500_000, 1_000_000]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def brute_force_knn(data: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Returns ground-truth indices (n_queries, k) by squared L2 — same metric as HNSW."""
    # Process in chunks to stay within memory for large N
    chunk = 50
    all_ids = []
    for start in range(0, len(queries), chunk):
        q = queries[start:start + chunk]          # (chunk, dim)
        # ||q - d||^2 = ||q||^2 + ||d||^2 - 2 q·d
        q2 = (q * q).sum(axis=1, keepdims=True)   # (chunk, 1)
        d2 = (data * data).sum(axis=1)             # (N,)
        dists = q2 + d2 - 2.0 * (q @ data.T)      # (chunk, N)
        idx = np.argpartition(dists, k, axis=1)[:, :k]
        # sort within the top-k
        rows = np.arange(len(q))[:, None]
        top_dists = dists[rows, idx]
        order = np.argsort(top_dists, axis=1)
        all_ids.append(idx[rows, order])
    return np.vstack(all_ids)


def recall_at_k(hnsw_ids: list[list[int]], bf_ids: np.ndarray, k: int) -> float:
    hits = 0
    total = len(hnsw_ids) * k
    for i, h_ids in enumerate(hnsw_ids):
        bf_set = set(bf_ids[i])
        hits += sum(1 for x in h_ids[:k] if x in bf_set)
    return hits / total


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark():
    rng = np.random.default_rng(SEED)

    results = {
        "sizes":           [],
        "build_times":     [],
        "hnsw_qps":        [],
        "bf_qps":          [],
        "recall":          [],
    }

    for N in DATASET_SIZES:
        ef = ef_for_size(N)
        print(f"\n{'='*55}")
        print(f"  N = {N:,}  |  dim={DIM}  M={M}  ef_construction={EF_CONST}  ef={ef}")
        print(f"{'='*55}")

        data    = rng.random((N,   DIM), dtype=np.float32)
        queries = rng.random((N_QUERY, DIM), dtype=np.float32)

        # --- Build ---
        idx = hnsw_index.HNSWIndex(dim=DIM, M=M, ef_construction=EF_CONST, seed=SEED)
        t0 = time.perf_counter()
        for v in data:
            idx.insert(v)
        build_time = time.perf_counter() - t0
        print(f"  Build:  {build_time:.2f}s  ({N/build_time:,.0f} inserts/s)")

        # --- HNSW query throughput ---
        hnsw_ids_all: list[list[int]] = []
        t0 = time.perf_counter()
        for q in queries:
            hits = idx.search(q, k=K, ef=ef)
            hnsw_ids_all.append([h[1] for h in hits])
        hnsw_time = time.perf_counter() - t0
        hnsw_qps  = N_QUERY / hnsw_time
        print(f"  HNSW:   {hnsw_qps:,.0f} queries/s  (latency {hnsw_time/N_QUERY*1000:.2f} ms/q)")

        # --- Brute-force throughput (per-query, not batched — fair comparison) ---
        bf_ids_list = []
        t0 = time.perf_counter()
        for q in queries:
            ids = brute_force_knn(data, q[np.newaxis], K)
            bf_ids_list.append(ids[0])
        bf_time = time.perf_counter() - t0
        bf_ids  = np.array(bf_ids_list)
        bf_qps  = N_QUERY / bf_time
        print(f"  BF:     {bf_qps:,.0f} queries/s  (latency {bf_time/N_QUERY*1000:.2f} ms/q)")

        # --- Recall ---
        rec = recall_at_k(hnsw_ids_all, bf_ids, K)
        print(f"  Recall@{K}: {rec:.4f}  (speedup {hnsw_qps/bf_qps:.1f}x)")

        results["sizes"].append(N)
        results["build_times"].append(build_time)
        results["hnsw_qps"].append(hnsw_qps)
        results["bf_qps"].append(bf_qps)
        results["recall"].append(rec)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: dict):
    sizes = results["sizes"]

    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"HNSW vs Brute-Force  |  dim={DIM}  M={M}  ef_construction={EF_CONST}  k={K}  (ef scaled per size)",
        fontsize=13, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # -- 1. Query throughput --
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(sizes, results["hnsw_qps"], "o-", color="#2196F3", label="HNSW", linewidth=2, markersize=6)
    ax1.plot(sizes, results["bf_qps"],   "s--", color="#F44336", label="Brute-force", linewidth=2, markersize=6)
    ax1.set_title("Query Throughput")
    ax1.set_xlabel("Dataset size (vectors)")
    ax1.set_ylabel("Queries / second")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # -- 2. Recall@K --
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sizes, [r * 100 for r in results["recall"]], "o-", color="#4CAF50", linewidth=2, markersize=6)
    ax2.axhline(y=90, color="gray", linestyle=":", alpha=0.6, label="90% target")
    ax2.set_title(f"Recall@{K}")
    ax2.set_xlabel("Dataset size (vectors)")
    ax2.set_ylabel("Recall (%)")
    ax2.set_xscale("log")
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # -- 3. Build time --
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(sizes, results["build_times"], "o-", color="#FF9800", linewidth=2, markersize=6)
    ax3.set_title("HNSW Build Time")
    ax3.set_xlabel("Dataset size (vectors)")
    ax3.set_ylabel("Seconds")
    ax3.set_xscale("log")
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # -- 4. Speedup --
    ax4 = fig.add_subplot(gs[1, 1])
    speedups = [h / b for h, b in zip(results["hnsw_qps"], results["bf_qps"])]
    ax4.bar(range(len(sizes)), speedups, color="#9C27B0", alpha=0.8)
    ax4.set_title("HNSW Speedup vs Brute-Force")
    ax4.set_xlabel("Dataset size")
    ax4.set_ylabel("Speedup (×)")
    ax4.set_xticks(range(len(sizes)))
    ax4.set_xticklabels([f"{s:,}" for s in sizes], rotation=30, ha="right")
    ax4.axhline(y=1, color="black", linestyle="-", linewidth=0.8)
    ax4.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(speedups):
        ax4.text(i, v + 0.05, f"{v:.1f}×", ha="center", va="bottom", fontsize=9)

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")
    return out_path


if __name__ == "__main__":
    results  = run_benchmark()
    out_path = plot_results(results)

    print("\n" + "="*55)
    print("Summary")
    print("="*55)
    print(f"{'Size':>10}  {'HNSW QPS':>12}  {'BF QPS':>12}  {'Recall@10':>10}  {'Speedup':>8}")
    print("-"*55)
    for i, N in enumerate(results["sizes"]):
        print(
            f"{N:>10,}  "
            f"{results['hnsw_qps'][i]:>12,.0f}  "
            f"{results['bf_qps'][i]:>12,.0f}  "
            f"{results['recall'][i]:>10.4f}  "
            f"{results['hnsw_qps'][i]/results['bf_qps'][i]:>7.1f}×"
        )
