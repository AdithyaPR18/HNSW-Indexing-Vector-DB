#pragma once

#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cmath>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <algorithm>
#include <functional>

// Candidate element for priority queues: (distance, node_id)
using Candidate = std::pair<float, int>;

// Max-heap comparator (furthest first)
struct MaxHeapCmp {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.first < b.first;
    }
};

// Min-heap comparator (nearest first)
struct MinHeapCmp {
    bool operator()(const Candidate& a, const Candidate& b) const {
        return a.first > b.first;
    }
};

using MaxHeap = std::priority_queue<Candidate, std::vector<Candidate>, MaxHeapCmp>;
using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>, MinHeapCmp>;

class HNSWIndex {
public:
    // M          : max connections per node per layer (paper recommends 16)
    // ef_construction: beam width during construction (paper recommends 200)
    // dim        : vector dimensionality
    // seed       : rng seed for reproducibility
    HNSWIndex(int dim, int M = 16, int ef_construction = 200, unsigned int seed = 42);

    // Insert a vector; returns the assigned internal id
    int insert(const std::vector<float>& vec);

    // k-NN search, returns (distance, id) pairs sorted nearest-first
    std::vector<Candidate> search(const std::vector<float>& query, int k, int ef = -1) const;

    // Mark a node deleted (lazy deletion — excluded from future results)
    void remove(int id);

    int size() const { return static_cast<int>(vectors_.size()); }
    int dim()  const { return dim_; }

private:
    // ---- graph storage ----
    // neighbors_[layer][node] = list of neighbour ids
    std::vector<std::unordered_map<int, std::vector<int>>> neighbors_;
    std::vector<std::vector<float>> vectors_;    // raw vectors, indexed by node id
    std::unordered_set<int> deleted_;
    int entry_point_ = -1;
    int max_layer_   = -1;

    // ---- hyper-params ----
    int dim_;
    int M_;              // max connections at layers 1+
    int M0_;             // max connections at layer 0  (= 2*M per paper)
    int ef_construction_;
    double m_L_;         // normalisation factor = 1 / ln(M)

    mutable std::mutex mtx_;
    mutable std::mt19937 rng_;

    // ---- helpers ----
    float distance(const std::vector<float>& a, const std::vector<float>& b) const;

    int random_level();

    // Greedy search from ep down to target_layer; returns ef-best candidates
    MaxHeap search_layer(const std::vector<float>& query,
                         int ep,
                         int ef,
                         int layer) const;

    // SELECT-NEIGHBORS-SIMPLE (Algorithm 3 in paper)
    std::vector<int> select_neighbors_simple(
        const std::vector<float>& query,
        MaxHeap candidates,
        int M) const;

    // SELECT-NEIGHBORS-HEURISTIC (Algorithm 4 in paper)
    // Keeps diverse neighbours; better recall than simple for M<16
    std::vector<int> select_neighbors_heuristic(
        const std::vector<float>& query,
        MaxHeap candidates,
        int M,
        int layer,
        bool extend_candidates = false,
        bool keep_pruned = true) const;

    void ensure_layer(int layer);
};
