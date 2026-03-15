#include <gtest/gtest.h>
#include "distance.hpp"
#include <cmath>
#include <random>
#include <vector>

using namespace neuraldb;

static std::vector<float> rand_vec(size_t dim, int seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// ── cosine: naive vs AVX2 ──────────────────────────────────────────────────────
class DistanceTest : public ::testing::TestWithParam<size_t> {};

TEST_P(DistanceTest, CosineAVX2MatchesNaive) {
    size_t dim = GetParam();
    for (int trial = 0; trial < 100; ++trial) {
        auto a = rand_vec(dim, trial);
        auto b = rand_vec(dim, trial + 1000);
        float naive  = cosine_similarity_naive(a.data(), b.data(), dim);
        float avx2   = cosine_similarity_avx2 (a.data(), b.data(), dim);
        EXPECT_NEAR(naive, avx2, 1e-5f)
            << "dim=" << dim << " trial=" << trial;
    }
}

TEST_P(DistanceTest, L2AVX2MatchesNaive) {
    size_t dim = GetParam();
    for (int trial = 0; trial < 100; ++trial) {
        auto a = rand_vec(dim, trial + 2000);
        auto b = rand_vec(dim, trial + 3000);
        float naive = l2_distance_naive(a.data(), b.data(), dim);
        float avx2  = l2_distance_avx2 (a.data(), b.data(), dim);
        // L2 can be large — use relative tolerance
        float tol = std::max(1e-4f, std::abs(naive) * 1e-4f);
        EXPECT_NEAR(naive, avx2, tol)
            << "dim=" << dim << " trial=" << trial;
    }
}

INSTANTIATE_TEST_SUITE_P(Dimensions, DistanceTest,
    ::testing::Values(1, 7, 8, 9, 16, 31, 32, 63, 64, 128, 256, 768, 1536));

// ── cosine: self-similarity = 1 ───────────────────────────────────────────────
TEST(DistanceSanity, CosineSelfIsOne) {
    auto v = rand_vec(768);
    EXPECT_NEAR(cosine_similarity_avx2(v.data(), v.data(), v.size()), 1.f, 1e-5f);
    EXPECT_NEAR(cosine_similarity_naive(v.data(), v.data(), v.size()), 1.f, 1e-5f);
}

// ── zero vector ───────────────────────────────────────────────────────────────
TEST(DistanceSanity, ZeroVector) {
    std::vector<float> a(128, 0.f);
    auto b = rand_vec(128);
    EXPECT_EQ(cosine_similarity_avx2(a.data(), b.data(), 128), 0.f);
    EXPECT_EQ(cosine_similarity_naive(a.data(), b.data(), 128), 0.f);
}

// ── L2: identical vectors ─────────────────────────────────────────────────────
TEST(DistanceSanity, L2SelfIsZero) {
    auto v = rand_vec(768);
    EXPECT_NEAR(l2_distance_avx2(v.data(), v.data(), v.size()), 0.f, 1e-6f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
