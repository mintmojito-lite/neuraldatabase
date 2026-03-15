// Scalar (naive) cosine similarity — reference implementation
#include "distance.hpp"
#include <cmath>

namespace neuraldb {

float cosine_similarity_naive(const float* a, const float* b, size_t dim) {
    double dot  = 0.0;
    double normA = 0.0;
    double normB = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        dot   += static_cast<double>(a[i]) * b[i];
        normA += static_cast<double>(a[i]) * a[i];
        normB += static_cast<double>(b[i]) * b[i];
    }
    const double denom = std::sqrt(normA) * std::sqrt(normB);
    if (denom < 1e-12) return 0.f;
    return static_cast<float>(dot / denom);
}

float l2_distance_naive(const float* a, const float* b, size_t dim) {
    double acc = 0.0;
    for (size_t i = 0; i < dim; ++i) {
        double d = static_cast<double>(a[i]) - b[i];
        acc += d * d;
    }
    return static_cast<float>(acc);
}

} // namespace neuraldb
