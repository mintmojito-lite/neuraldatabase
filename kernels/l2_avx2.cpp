// AVX2-accelerated squared L2 distance
// L2(a,b) = sum((a[i]-b[i])^2)
#include "distance.hpp"
#include <immintrin.h>
#include <cmath>

namespace neuraldb {

static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float l2_distance_avx2(const float* a, const float* b, size_t dim) {
    __m256 acc = _mm256_setzero_ps();

    size_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        __m256 va   = _mm256_loadu_ps(a + i);
        __m256 vb   = _mm256_loadu_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        acc = _mm256_fmadd_ps(diff, diff, acc);
    }

    float result = hsum256(acc);

    // Scalar tail
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }

    return result;
}

} // namespace neuraldb
