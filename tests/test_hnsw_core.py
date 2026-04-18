"""
Step-1 smoke test:
  Insert 1000 random 128-dim vectors into the HNSW index,
  query each one as its own nearest neighbour,
  verify the result against brute-force cosine / L2 search.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import hnsw_index


def brute_force_knn(data: np.ndarray, query: np.ndarray, k: int):
    """Return (distances, indices) sorted nearest-first (squared L2)."""
    diffs = data - query
    dists = (diffs * diffs).sum(axis=1)
    idx = np.argsort(dists)[:k]
    return dists[idx], idx


def recall_at_k(hnsw_ids, bf_ids, k):
    hnsw_set = set(hnsw_ids[:k])
    bf_set   = set(bf_ids[:k])
    return len(hnsw_set & bf_set) / k


def main():
    rng   = np.random.default_rng(0)
    N     = 1000
    DIM   = 128
    K     = 10
    EF    = 64   # search beam width

    print(f"Building HNSW index with N={N}, dim={DIM}, M=16, ef_construction=200")
    idx = hnsw_index.HNSWIndex(dim=DIM, M=16, ef_construction=200, seed=42)

    vecs = rng.random((N, DIM), dtype=np.float32)
    for i, v in enumerate(vecs):
        idx.insert(v)

    assert len(idx) == N, f"Expected {N} elements, got {len(idx)}"
    print(f"  Inserted {len(idx)} vectors. ✓")

    # Query all N vectors and measure recall@K
    recalls = []
    for i in range(N):
        q = vecs[i]
        hnsw_results = idx.search(q, k=K, ef=EF)
        hnsw_ids     = [r[1] for r in hnsw_results]

        _, bf_ids    = brute_force_knn(vecs, q, k=K)

        recalls.append(recall_at_k(hnsw_ids, bf_ids, K))

    mean_recall = np.mean(recalls)
    print(f"  Mean recall@{K} over {N} queries: {mean_recall:.4f}")
    assert mean_recall >= 0.90, f"Recall too low: {mean_recall:.4f} (expected ≥ 0.90)"
    print(f"  Recall ≥ 0.90  ✓")

    # NN of itself should always be itself (distance = 0)
    print("  Checking self-query (nearest neighbour of each vector is itself)...")
    failures = 0
    for i in range(N):
        results = idx.search(vecs[i], k=1, ef=EF)
        if not results or results[0][1] != i:
            failures += 1
    print(f"  Self-query failures: {failures}/{N}")
    assert failures == 0, f"{failures} self-query failures"
    print(f"  Self-query  ✓")

    # Delete test
    print("  Testing lazy deletion...")
    idx.remove(0)
    results = idx.search(vecs[0], k=1, ef=EF)
    if results:
        assert results[0][1] != 0, "Deleted node 0 still appears in results"
    print(f"  Deleted node excluded from results  ✓")

    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
