#pragma once
#include "distance.hpp"
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace neuraldb {

struct Vector {
    uint64_t           id;
    std::string        metadata;
    std::vector<float> data;
    bool               deleted = false;

    const float* ptr() const noexcept { return data.data(); }
    const float* data_ptr() const noexcept { return data.data(); }
    float operator[](size_t i) const { return data[i]; }
    float& operator[](size_t i) { return data[i]; }
};

struct SearchResult {
    uint64_t id;
    float score;
    std::string metadata;
};

/// Flat brute-force index. O(N) search, OpenMP-parallelised.
class FlatIndex {
public:
  explicit FlatIndex(size_t dim) : dim_(dim) {}

  void insert(uint64_t id, const float *vec, std::string metadata = "");
  void insert(uint64_t id, const std::vector<float> &vec,
              std::string metadata = "");

  /// Soft-delete by external id.
  void delete_vector(uint64_t id);

  /// Remove deleted entries from vectors_ (in-place).
  void compact();

  std::vector<SearchResult> search(const float *query, int k) const;
  std::vector<SearchResult> search(const std::vector<float> &query,
                                    int k) const;

  std::vector<SearchResult> search_filtered(
      const float *query, int k,
      const std::function<bool(const std::string &)> &filter_fn) const;

  size_t size() const noexcept { return vectors_.size(); }
  size_t live_size() const noexcept {
      size_t c = 0; for (auto& v : vectors_) if (!v.deleted) ++c; return c;
  }
  size_t dim() const noexcept { return dim_; }

  const std::vector<Vector> &raw_vectors() const noexcept { return vectors_; }

private:
  size_t dim_;
  std::vector<Vector> vectors_;
};

} // namespace neuraldb
