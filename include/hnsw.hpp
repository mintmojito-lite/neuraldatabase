#pragma once
#include <shared_mutex>
#include <atomic>
#include <functional>
#include <vector>
#include <string>
#include <cstdint>

namespace neuraldb {

struct SearchResult;  // forward declared in vector_store.hpp

struct HNSWConfig {
    int    M              = 16;
    int    ef_construction = 200;
    int    ef_search       = 50;
    // mL = 1/ln(M) — layer assignment multiplier
};

/// HNSW approximate nearest-neighbour index.
/// Implements Malkov & Yashunin 2016 (arXiv:1603.09320).
class HNSWIndex {
public:
    HNSWIndex(size_t dim, HNSWConfig cfg = {});
    ~HNSWIndex();

    void insert(uint64_t id, const float* vec, std::string metadata = "");
    void insert(uint64_t id, const std::vector<float>& vec, std::string metadata = "");

    /// Soft-delete a vector by external ID.  O(N) id lookup.
    /// The vector stays in raw_vectors_; it is just tombstoned.
    void delete_vector(uint64_t external_id);

    /// Returns true if the external_id has been soft-deleted.
    bool is_deleted(uint64_t external_id) const;

    /// Returns total number of (non-compact) slots, including deleted ones.
    size_t size() const noexcept { return num_vectors_.load(std::memory_order_relaxed); }

    /// Returns number of live (non-deleted) vectors.
    size_t live_size() const noexcept;

    /// Rebuild the index keeping only live vectors.
    /// Saves the compacted index to index_path; checkpoints WAL at wal_path.
    void compact(const std::string& index_path, const std::string& wal_path);

    // Returns top-k approximate nearest neighbours (skips tombstoned)
    std::vector<std::pair<float, uint64_t>> search(
        const float* query, int k) const;
    std::vector<std::pair<float, uint64_t>> search(
        const std::vector<float>& query, int k) const;

    void save(const std::string& path) const;
    void load(const std::string& path);

    size_t dim()  const noexcept { return dim_; }
    void   set_ef_search(int ef) noexcept { cfg_.ef_search = ef; }

    const float* raw_vec(uint32_t internal_id) const noexcept {
        return raw_vectors_.data() + internal_id * dim_;
    }
    uint64_t external_id(uint32_t internal_id) const noexcept {
        return id_map_[internal_id];
    }
    const std::string& metadata(uint32_t internal_id) const noexcept {
        return metadata_[internal_id];
    }

    /// Override the vector accessor used during search (for PagedIndex).
    /// If unset, defaults to raw_vec().
    void set_vector_accessor(std::function<const float*(uint32_t)> fn) {
        vec_accessor_ = std::move(fn);
    }

private:
    /// Resolve the vector for internal_id using the (possibly custom) accessor.
    const float* get_vec_(uint32_t internal_id) const noexcept {
        if (vec_accessor_) return vec_accessor_(internal_id);
        return raw_vectors_.data() + internal_id * dim_;
    }

    // ── search internals ─────────────────────────────────────────────────────
    /// Greedy beam search on one layer. Returns ef closest internal IDs.
    std::vector<std::pair<float,uint32_t>> search_layer_(
        const float* query, uint32_t entry_point,
        int ef, int layer) const;

    // ── neighbour selection ───────────────────────────────────────────────────
    /// Simple distance-based pruning: keep best M.
    std::vector<uint32_t> select_neighbours_(
        uint32_t q, const std::vector<std::pair<float,uint32_t>>& candidates,
        int M) const;

    // ── data ─────────────────────────────────────────────────────────────────
    mutable std::shared_mutex index_mutex_;

    size_t     dim_;
    HNSWConfig cfg_;
    float      mL_;           // = 1.0 / std::log(M)

    std::atomic<size_t> num_vectors_{0};
    int      max_layer_   = 0;
    uint32_t entry_point_ = 0;

    // node → layer → neighbour list (internal IDs)
    std::vector<std::vector<std::vector<uint32_t>>> graph_;
    // flat float storage: internal_id * dim_ → vector
    std::vector<float>       raw_vectors_;
    std::vector<uint64_t>    id_map_;         // internal → external
    std::vector<std::string> metadata_;

    // Tombstone bitset: tombstone_[internal_id] = true → soft-deleted
    std::vector<bool>        tombstone_;

    // Optional custom vector accessor (used by PagedIndex)
    std::function<const float*(uint32_t)> vec_accessor_;
};

} // namespace neuraldb
