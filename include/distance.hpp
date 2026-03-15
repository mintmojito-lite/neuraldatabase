#pragma once
#include <cstddef>

namespace neuraldb {

/// Scalar reference implementation — correct but not fast.
float cosine_similarity_naive(const float* a, const float* b, size_t dim);

/// AVX2 accelerated cosine similarity — processes 8 floats/cycle.
float cosine_similarity_avx2(const float* a, const float* b, size_t dim);

/// ARM NEON cosine similarity — processes 4 floats/cycle.
/// On x86, falls through to AVX2.
float cosine_similarity_neon(const float* a, const float* b, size_t dim);

/// Runtime-dispatched cosine similarity.
/// ARM → NEON  |  x86 → AVX2  |  else → scalar
float cosine_similarity(const float* a, const float* b, size_t dim);

/// AVX2 accelerated squared Euclidean (L2) distance.
float l2_distance_avx2(const float* a, const float* b, size_t dim);

/// ARM NEON squared L2 distance.
/// On x86, falls through to AVX2.
float l2_distance_neon(const float* a, const float* b, size_t dim);

/// Runtime-dispatched squared L2 distance.
float l2_distance(const float* a, const float* b, size_t dim);

/// L2 scalar reference (for testing).
float l2_distance_naive(const float* a, const float* b, size_t dim);

} // namespace neuraldb
