/// test_neon.cpp
/// Task 2: ARM NEON runtime dispatch — validates accuracy & dispatch.
/// On ARM: NEON vs scalar accuracy within 1e-5.
/// On x86: validates AVX2 dispatch (NEON stub calls AVX2).
#include <gtest/gtest.h>
#include "distance.hpp"
#include <cmath>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>

using namespace neuraldb;

static std::vector<float> rand_vec(size_t dim, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    float n = 0.f; for (auto x : v) n += x*x; n = std::sqrtf(n);
    for (auto& x : v) x /= n;
    return v;
}

// ── Accuracy tests ────────────────────────────────────────────────────────────

template<size_t DIM>
void check_cosine_neon_vs_naive(int seed_a, int seed_b, float tol = 1e-5f) {
    auto a = rand_vec(DIM, seed_a);
    auto b = rand_vec(DIM, seed_b);
    float ref  = cosine_similarity_naive(a.data(), b.data(), DIM);
    float neon = cosine_similarity_neon(a.data(), b.data(), DIM);
    EXPECT_NEAR(neon, ref, tol)
        << "cosine_similarity_neon dim=" << DIM << " mismatch";
}

template<size_t DIM>
void check_l2_neon_vs_naive(int seed_a, int seed_b, float tol = 1e-4f) {
    auto a = rand_vec(DIM, seed_a);
    auto b = rand_vec(DIM, seed_b);
    float ref  = l2_distance_naive(a.data(), b.data(), DIM);
    float neon = l2_distance_neon(a.data(), b.data(), DIM);
    EXPECT_NEAR(neon, ref, tol)
        << "l2_distance_neon dim=" << DIM << " mismatch";
}

TEST(NEON, CosineAccuracy_128)  { for (int i=0;i<10;++i) check_cosine_neon_vs_naive<128>(i, i+100); }
TEST(NEON, CosineAccuracy_256)  { for (int i=0;i<10;++i) check_cosine_neon_vs_naive<256>(i, i+200); }
TEST(NEON, CosineAccuracy_384)  { for (int i=0;i<10;++i) check_cosine_neon_vs_naive<384>(i, i+300); }
TEST(NEON, CosineAccuracy_768)  { for (int i=0;i<10;++i) check_cosine_neon_vs_naive<768>(i, i+400); }

TEST(NEON, L2Accuracy_128)  { for (int i=0;i<10;++i) check_l2_neon_vs_naive<128>(i, i+100); }
TEST(NEON, L2Accuracy_256)  { for (int i=0;i<10;++i) check_l2_neon_vs_naive<256>(i, i+200); }
TEST(NEON, L2Accuracy_384)  { for (int i=0;i<10;++i) check_l2_neon_vs_naive<384>(i, i+300); }
TEST(NEON, L2Accuracy_768)  { for (int i=0;i<10;++i) check_l2_neon_vs_naive<768>(i, i+400); }

// ── Dispatch test ─────────────────────────────────────────────────────────────

TEST(NEON, DispatchMatchesBestKernel) {
    // cosine_similarity() (dispatch) should match the best available kernel
    for (size_t dim : {128u, 256u, 384u, 768u}) {
        auto a = rand_vec(dim, 42);
        auto b = rand_vec(dim, 99);
        float dispatched = cosine_similarity(a.data(), b.data(), dim);
        float naive      = cosine_similarity_naive(a.data(), b.data(), dim);
        EXPECT_NEAR(dispatched, naive, 1e-5f) << "dispatch mismatch at dim=" << dim;
    }
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

TEST(NEON, BenchmarkVsScalar) {
    using Clock = std::chrono::high_resolution_clock;
    const int ITERS = 100000;

    for (size_t dim : {128u, 256u, 384u, 768u}) {
        auto a = rand_vec(dim, 1);
        auto b = rand_vec(dim, 2);

        auto t0 = Clock::now();
        volatile float s = 0.f;
        for (int i = 0; i < ITERS; ++i)
            s = cosine_similarity_naive(a.data(), b.data(), dim);
        auto t1 = Clock::now();
        for (int i = 0; i < ITERS; ++i)
            s = cosine_similarity_neon(a.data(), b.data(), dim);
        auto t2 = Clock::now();
        (void)s;

        double scalar_ns = std::chrono::duration<double,std::nano>(t1-t0).count() / ITERS;
        double neon_ns   = std::chrono::duration<double,std::nano>(t2-t1).count() / ITERS;
        std::cout << "[NEON vs Scalar] dim=" << dim
                  << "  scalar=" << scalar_ns << "ns"
#ifdef __ARM_NEON
                  << "  neon=" << neon_ns << "ns"
                  << "  speedup=" << scalar_ns/neon_ns << "x\n";
#else
                  << "  neon(AVX2_stub)=" << neon_ns << "ns\n";
#endif
    }
    // No assertion: benchmark is informational
    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
