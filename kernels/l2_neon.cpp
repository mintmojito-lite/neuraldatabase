// ARM NEON squared L2 distance.
// On ARM: processes 4 floats/cycle with vfmaq_f32.
// On x86: stub that routes to AVX2.
#include "distance.hpp"

#ifdef __ARM_NEON
#include <arm_neon.h>

namespace neuraldb {

float l2_distance_neon(const float* a, const float* b, size_t dim) {
    float32x4_t acc = vdupq_n_f32(0.f);

    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va   = vld1q_f32(a + i);
        float32x4_t vb   = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }

    float result = vaddvq_f32(acc);

    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        result += d * d;
    }
    return result;
}

} // namespace neuraldb

#else
#include "distance.hpp"
namespace neuraldb {
float l2_distance_neon(const float* a, const float* b, size_t dim) {
    return l2_distance_avx2(a, b, dim);
}
} // namespace neuraldb
#endif
