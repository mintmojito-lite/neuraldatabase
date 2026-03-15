#include "distance.hpp"

namespace neuraldb {

// Runtime dispatch wrappers
float cosine_similarity(const float* a, const float* b, size_t dim) {
#ifdef __ARM_NEON
    return cosine_similarity_neon(a, b, dim);
#elif defined(__AVX2__)
    return cosine_similarity_avx2(a, b, dim);
#else
    return cosine_similarity_naive(a, b, dim);
#endif
}

float l2_distance(const float* a, const float* b, size_t dim) {
#ifdef __ARM_NEON
    return l2_distance_neon(a, b, dim);
#elif defined(__AVX2__)
    return l2_distance_avx2(a, b, dim);
#else
    return l2_distance_naive(a, b, dim);
#endif
}

} // namespace neuraldb
