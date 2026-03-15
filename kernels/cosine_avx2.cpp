// AVX2-accelerated cosine similarity
// Processes 8 floats per cycle using _mm256_fmadd_ps.
// Handles tail elements (dim % 8 != 0) with scalar loop.
#include "distance.hpp"
#include <immintrin.h>
#include <cmath>

namespace neuraldb {

// Horizontal sum of __m256 (sum all 8 lanes)
static inline float hsum256(__m256 v) {
    // add high/low 128-bit halves
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    // horizontal add within 128-bit
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float cosine_similarity_avx2(const float* a, const float* b, size_t dim) {
    __m256 dot_acc  = _mm256_setzero_ps();
    __m256 normA_acc = _mm256_setzero_ps();
    __m256 normB_acc = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        dot_acc   = _mm256_fmadd_ps(va, vb, dot_acc);
        normA_acc = _mm256_fmadd_ps(va, va, normA_acc);
        normB_acc = _mm256_fmadd_ps(vb, vb, normB_acc);
    }

    float dot  = hsum256(dot_acc);
    float normA = hsum256(normA_acc);
    float normB = hsum256(normB_acc);

    // Scalar tail
    for (; i < dim; ++i) {
        dot   += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    float denom = std::sqrt(normA) * std::sqrt(normB);
    if (denom < 1e-12f) return 0.f;
    return dot / denom;
}

} // namespace neuraldb
