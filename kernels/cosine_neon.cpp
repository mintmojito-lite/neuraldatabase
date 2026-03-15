// ARM NEON cosine similarity.
// On ARM: uses vfmaq_f32 to process 4 floats/cycle.
// On x86: defined as stubs that route to AVX2 — never called directly.
#include "distance.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>
#include <cmath>

namespace neuraldb {

float cosine_similarity_neon(const float* a, const float* b, size_t dim) {
    float32x4_t dot_acc   = vdupq_n_f32(0.f);
    float32x4_t normA_acc = vdupq_n_f32(0.f);
    float32x4_t normB_acc = vdupq_n_f32(0.f);

    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        dot_acc   = vfmaq_f32(dot_acc,   va, vb);
        normA_acc = vfmaq_f32(normA_acc, va, va);
        normB_acc = vfmaq_f32(normB_acc, vb, vb);
    }

    // Horizontal reduction
    float dot   = vaddvq_f32(dot_acc);
    float normA = vaddvq_f32(normA_acc);
    float normB = vaddvq_f32(normB_acc);

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

#else
// x86 fallback — dispatch handled in distance.cpp
#include "distance.hpp"
namespace neuraldb {
float cosine_similarity_neon(const float* a, const float* b, size_t dim) {
    return cosine_similarity_avx2(a, b, dim);
}
} // namespace neuraldb
#endif
