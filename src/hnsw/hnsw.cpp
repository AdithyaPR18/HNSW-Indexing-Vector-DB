#include "hnsw.h"
#include <cassert>
#include <numeric>

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

HNSWIndex::HNSWIndex(int dim, int M, int ef_construction, unsigned int seed)
    : dim_(dim),
      M_(M),
      M0_(2 * M),
      ef_construction_(ef_construction),
      m_L_(1.0 / std::log(static_cast<double>(M))),
      rng_(seed)
{}

// ---------------------------------------------------------------------------
// Distance  (squared Euclidean — monotone so ordering is preserved)
// ---------------------------------------------------------------------------

float HNSWIndex::distance(const std::vector<float>& a,
                          const std::vector<float>& b) const {
    float sum = 0.0f;
    for (int i = 0; i < dim_; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

// ---------------------------------------------------------------------------
// Level sampling  (Algorithm 1, line 4)
// ---------------------------------------------------------------------------

int HNSWIndex::random_level() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    int level = static_cast<int>(-std::log(dist(rng_)) * m_L_);
    return level;
}

// ---------------------------------------------------------------------------
// Layer bookkeeping
// ---------------------------------------------------------------------------

void HNSWIndex::ensure_layer(int layer) {
    while (static_cast<int>(neighbors_.size()) <= layer) {
        neighbors_.emplace_back();
    }
}

// ---------------------------------------------------------------------------
// Search within one layer  (Algorithm 2)
// Returns a MAX-heap of size ef, containing the ef closest candidates found.
// ---------------------------------------------------------------------------

MaxHeap HNSWIndex::search_layer(const std::vector<float>& query,
                                int ep,
                                int ef,
                                int layer) const {
    // visited set
    std::unordered_set<int> visited;
    visited.insert(ep);

    float ep_dist = distance(query, vectors_[ep]);

    // W  = dynamic candidate list (max-heap, furthest on top)
    MaxHeap W;
    W.emplace(ep_dist, ep);

    // C  = candidates to explore (min-heap, nearest on top)
    MinHeap C;
    C.emplace(ep_dist, ep);

    while (!C.empty()) {
        auto [c_dist, c] = C.top();
        C.pop();

        float f_dist = W.top().first; // furthest in result set

        if (c_dist > f_dist && static_cast<int>(W.size()) >= ef) {
            // All remaining candidates are further than our worst result — stop
            break;
        }

        // Explore neighbours of c at this layer
        const auto& layer_map = neighbors_[layer];
        auto it = layer_map.find(c);
        if (it == layer_map.end()) continue;

        for (int e : it->second) {
            if (deleted_.count(e)) continue;
            if (!visited.insert(e).second) continue;

            float e_dist = distance(query, vectors_[e]);
            float f_dist2 = W.top().first;

            if (static_cast<int>(W.size()) < ef || e_dist < f_dist2) {
                C.emplace(e_dist, e);
                W.emplace(e_dist, e);
                if (static_cast<int>(W.size()) > ef) {
                    W.pop(); // remove furthest
                }
            }
        }
    }

    return W;
}

// ---------------------------------------------------------------------------
// SELECT-NEIGHBORS-SIMPLE  (Algorithm 3)
// ---------------------------------------------------------------------------

std::vector<int> HNSWIndex::select_neighbors_simple(
        const std::vector<float>& /*query*/,
        MaxHeap candidates,
        int M) const {

    // Convert max-heap → sorted nearest-first
    std::vector<Candidate> all;
    all.reserve(candidates.size());
    while (!candidates.empty()) {
        all.push_back(candidates.top());
        candidates.pop();
    }
    std::sort(all.begin(), all.end(), [](const Candidate& a, const Candidate& b) {
        return a.first < b.first;
    });

    std::vector<int> result;
    result.reserve(std::min(M, static_cast<int>(all.size())));
    for (int i = 0; i < M && i < static_cast<int>(all.size()); ++i) {
        result.push_back(all[i].second);
    }
    return result;
}

// ---------------------------------------------------------------------------
// SELECT-NEIGHBORS-HEURISTIC  (Algorithm 4)
// Produces diverse neighbours, improving recall at the cost of slightly more
// work at construction time.
// ---------------------------------------------------------------------------

std::vector<int> HNSWIndex::select_neighbors_heuristic(
        const std::vector<float>& query,
        MaxHeap candidates,
        int M,
        int layer,
        bool extend_candidates,
        bool keep_pruned) const {

    // Flatten to min-heap so we process nearest-first
    MinHeap W_d; // working set
    while (!candidates.empty()) {
        W_d.emplace(candidates.top().first, candidates.top().second);
        candidates.pop();
    }

    if (extend_candidates) {
        // Optionally expand by neighbours of neighbours
        std::unordered_set<int> seen;
        std::vector<Candidate> existing;
        while (!W_d.empty()) {
            existing.push_back(W_d.top());
            seen.insert(W_d.top().second);
            W_d.pop();
        }
        for (auto& [d, e] : existing) {
            W_d.emplace(d, e);
            const auto& lm = neighbors_[layer];
            auto it = lm.find(e);
            if (it == lm.end()) continue;
            for (int adj : it->second) {
                if (deleted_.count(adj)) continue;
                if (!seen.insert(adj).second) continue;
                W_d.emplace(distance(query, vectors_[adj]), adj);
            }
        }
    }

    std::vector<int> R;             // result set
    MinHeap W_discarded;            // pruned but kept if keep_pruned

    while (!W_d.empty() && static_cast<int>(R.size()) < M) {
        auto [d_e, e] = W_d.top();
        W_d.pop();

        if (deleted_.count(e)) continue;

        // Accept e if it is closer to the query than to any already-selected neighbour
        bool good = true;
        for (int r : R) {
            if (distance(vectors_[e], vectors_[r]) < d_e) {
                good = false;
                break;
            }
        }
        if (good) {
            R.push_back(e);
        } else if (keep_pruned) {
            W_discarded.emplace(d_e, e);
        }
    }

    if (keep_pruned) {
        while (!W_discarded.empty() && static_cast<int>(R.size()) < M) {
            R.push_back(W_discarded.top().second);
            W_discarded.pop();
        }
    }

    return R;
}

// ---------------------------------------------------------------------------
// INSERT  (Algorithm 1)
// ---------------------------------------------------------------------------

int HNSWIndex::insert(const std::vector<float>& vec) {
    if (static_cast<int>(vec.size()) != dim_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }

    std::lock_guard<std::mutex> lock(mtx_);

    int q = static_cast<int>(vectors_.size());
    vectors_.push_back(vec);

    int l = random_level();
    ensure_layer(l);

    // Register q in every layer up to l
    for (int lc = 0; lc <= l; ++lc) {
        neighbors_[lc][q] = {};
    }

    if (entry_point_ == -1) {
        // First element
        entry_point_ = q;
        max_layer_   = l;
        return q;
    }

    int ep = entry_point_;
    int L  = max_layer_;

    // Greedy descent from L down to l+1 (find a good entry point for layer l)
    for (int lc = L; lc > l; --lc) {
        if (lc < static_cast<int>(neighbors_.size())) {
            MaxHeap candidates = search_layer(vec, ep, 1, lc);
            ep = candidates.top().second;
        }
    }

    // From min(l, L) down to 0: insert q with ef_construction beam
    for (int lc = std::min(l, L); lc >= 0; --lc) {
        MaxHeap W = search_layer(vec, ep, ef_construction_, lc);

        int M_lc = (lc == 0) ? M0_ : M_;
        std::vector<int> neighbours = select_neighbors_heuristic(vec, W, M_lc, lc);

        // Connect q → neighbours
        neighbors_[lc][q] = neighbours;

        // Connect neighbours → q, pruning if oversaturated
        for (int n : neighbours) {
            auto& n_links = neighbors_[lc][n];
            n_links.push_back(q);

            if (static_cast<int>(n_links.size()) > M_lc) {
                // Rebuild neighbour list via heuristic
                MaxHeap cands;
                for (int x : n_links) {
                    cands.emplace(distance(vectors_[n], vectors_[x]), x);
                }
                n_links = select_neighbors_heuristic(vectors_[n], cands, M_lc, lc);
            }
        }

        // Best candidate in W becomes next entry point
        if (!W.empty()) {
            // W is a max-heap; find the minimum (nearest) by draining it
            Candidate best = W.top();
            while (!W.empty()) {
                if (W.top().first < best.first) best = W.top();
                W.pop();
            }
            ep = best.second;
        }
    }

    if (l > L) {
        entry_point_ = q;
        max_layer_   = l;
    }

    return q;
}

// ---------------------------------------------------------------------------
// SEARCH  (Algorithm 5)
// ---------------------------------------------------------------------------

std::vector<Candidate> HNSWIndex::search(const std::vector<float>& query,
                                         int k,
                                         int ef) const {
    if (static_cast<int>(query.size()) != dim_) {
        throw std::invalid_argument("Query dimension mismatch");
    }
    if (ef < 0) ef = std::max(k, 50);

    std::lock_guard<std::mutex> lock(mtx_);

    if (entry_point_ == -1) return {};

    int ep = entry_point_;
    int L  = max_layer_;

    // Greedy descent from L down to layer 1
    for (int lc = L; lc > 0; --lc) {
        if (lc < static_cast<int>(neighbors_.size())) {
            MaxHeap W = search_layer(query, ep, 1, lc);
            ep = W.top().second;
        }
    }

    // Full ef-search at layer 0
    MaxHeap W = search_layer(query, ep, ef, 0);

    // Collect top-k nearest (W is max-heap, we want min-k)
    std::vector<Candidate> result;
    result.reserve(W.size());
    while (!W.empty()) {
        if (!deleted_.count(W.top().second)) {
            result.push_back(W.top());
        }
        W.pop();
    }

    // Sort nearest-first
    std::sort(result.begin(), result.end(), [](const Candidate& a, const Candidate& b) {
        return a.first < b.first;
    });

    if (static_cast<int>(result.size()) > k) result.resize(k);
    return result;
}

// ---------------------------------------------------------------------------
// REMOVE  (lazy deletion)
// ---------------------------------------------------------------------------

void HNSWIndex::remove(int id) {
    std::lock_guard<std::mutex> lock(mtx_);
    if (id < 0 || id >= static_cast<int>(vectors_.size())) {
        throw std::out_of_range("Node id out of range");
    }
    deleted_.insert(id);
}
