#include "vector_store.hpp"
#include "distance.hpp"
#include <algorithm>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace neuraldb {

void FlatIndex::insert(uint64_t id, const float* vec, std::string metadata) {
    vectors_.emplace_back();
    auto& v     = vectors_.back();
    v.id        = id;
    v.data.assign(vec, vec + dim_);
    v.metadata  = std::move(metadata);
    v.deleted   = false;
}

void FlatIndex::insert(uint64_t id, const std::vector<float>& vec, std::string metadata) {
    if (vec.size() != dim_) throw std::invalid_argument("vector dimension mismatch");
    insert(id, vec.data(), std::move(metadata));
}

void FlatIndex::delete_vector(uint64_t id) {
    for (auto& v : vectors_) {
        if (v.id == id) { v.deleted = true; return; }
    }
}

void FlatIndex::compact() {
    vectors_.erase(
        std::remove_if(vectors_.begin(), vectors_.end(),
                       [](const Vector& v){ return v.deleted; }),
        vectors_.end());
}

std::vector<SearchResult> FlatIndex::search(const float* query, int k) const {
    const int n = static_cast<int>(vectors_.size());
    if (n == 0) return {};

    std::vector<std::pair<float, int>> scores;
    scores.reserve(n);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        if (vectors_[i].deleted) continue;
#pragma omp critical
        scores.push_back({ cosine_similarity_avx2(query, vectors_[i].ptr(), dim_), i });
    }

    int actual_k = std::min(k, static_cast<int>(scores.size()));
    std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<SearchResult> results;
    results.reserve(actual_k);
    for (int i = 0; i < actual_k; ++i) {
        auto& [score, idx] = scores[i];
        results.push_back({ vectors_[idx].id, score, vectors_[idx].metadata });
    }
    return results;
}

std::vector<SearchResult> FlatIndex::search(const std::vector<float>& query, int k) const {
    if (query.size() != dim_) throw std::invalid_argument("query dimension mismatch");
    return search(query.data(), k);
}

std::vector<SearchResult> FlatIndex::search_filtered(
    const float* query, int k,
    const std::function<bool(const std::string&)>& filter_fn) const
{
    std::vector<std::pair<float, int>> scores;
    scores.reserve(vectors_.size());

    for (int i = 0; i < static_cast<int>(vectors_.size()); ++i) {
        if (vectors_[i].deleted) continue;
        if (filter_fn(vectors_[i].metadata)) {
            float s = cosine_similarity_avx2(query, vectors_[i].ptr(), dim_);
            scores.push_back({ s, i });
        }
    }

    int actual_k = std::min(k, static_cast<int>(scores.size()));
    if (actual_k == 0) return {};
    std::partial_sort(scores.begin(), scores.begin() + actual_k, scores.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    std::vector<SearchResult> results;
    results.reserve(actual_k);
    for (int i = 0; i < actual_k; ++i) {
        auto& [score, idx] = scores[i];
        results.push_back({ vectors_[idx].id, score, vectors_[idx].metadata });
    }
    return results;
}

} // namespace neuraldb
