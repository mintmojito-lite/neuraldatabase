/// test_paged.cpp
/// Task 3: Disk-based paging (PagedIndex) — correctness and basic benchmark.
#include <gtest/gtest.h>
#include "paged_index.hpp"
#include "hnsw.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

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

// ── Correctness ───────────────────────────────────────────────────────────────

TEST(PagedIndex, SearchMatchesInMemory) {
    const size_t N = 2000, DIM = 64;
    const std::string data_path = "test_paged_tmp.bin";

    // tiny cache: only 4 pages
    size_t tiny_cache = 4 * PAGED_PAGE_SIZE;

    HNSWConfig cfg; cfg.M = 12; cfg.ef_construction = 100; cfg.ef_search = 100;
    PagedIndex paged(data_path, DIM, tiny_cache, cfg);
    HNSWIndex  inmem(DIM, cfg);

    std::vector<std::vector<float>> vecs(N);
    for (size_t i = 0; i < N; ++i) {
        vecs[i] = rand_vec(DIM, static_cast<int>(i) + 5000);
        paged.insert(i, vecs[i]);
        inmem.insert(i, vecs[i]);
    }

    int matches = 0, queries = 20;
    for (int qi = 0; qi < queries; ++qi) {
        auto q = rand_vec(DIM, qi + 99999);
        auto pr = paged.search(q, 1);
        auto ir = inmem.search(q, 1);
        if (!pr.empty() && !ir.empty() && pr[0].second == ir[0].second)
            ++matches;
    }
    // Expect at least 75% top-1 agreement (HNSW is approximate + both share same graph seed)
    EXPECT_GE(matches, queries * 3 / 4)
        << "Too few top-1 matches between paged and in-memory";

    remove(data_path.c_str());
}

TEST(PagedIndex, LRUEvictionIsTransparent) {
    // Ensure the LRU cache evicts pages and reloads them correctly
    const size_t N = 500, DIM = 32;
    const std::string data_path = "test_paged_lru_tmp.bin";
    size_t tiny_cache = 2 * PAGED_PAGE_SIZE; // very small: 2 pages

    HNSWConfig cfg; cfg.M = 8; cfg.ef_construction = 50; cfg.ef_search = 50;
    PagedIndex paged(data_path, DIM, tiny_cache, cfg);

    for (size_t i = 0; i < N; ++i)
        paged.insert(i, rand_vec(DIM, static_cast<int>(i)));

    // Run 50 queries — should not crash or return garbage
    for (int qi = 0; qi < 50; ++qi) {
        auto q = rand_vec(DIM, qi + 12345);
        auto r = paged.search(q, 5);
        ASSERT_FALSE(r.empty()) << "paged search returned empty for qi=" << qi;
        for (auto& [score, id] : r) {
            EXPECT_GE(score, -1.f);
            EXPECT_LE(score, 1.f);
        }
    }
    remove(data_path.c_str());
}

// ── Benchmark ─────────────────────────────────────────────────────────────────

TEST(PagedIndex, BenchmarkPagedVsInMemory) {
    using Clock = std::chrono::high_resolution_clock;
    const size_t N = 5000, DIM = 128, K = 10;
    const std::string data_path = "test_paged_bench_tmp.bin";

    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 50;
    PagedIndex paged(data_path, DIM, 64 * PAGED_PAGE_SIZE, cfg); // 256KB cache
    HNSWIndex  inmem(DIM, cfg);

    for (size_t i = 0; i < N; ++i) {
        auto v = rand_vec(DIM, static_cast<int>(i));
        paged.insert(i, v);
        inmem.insert(i, v);
    }

    const int Q = 100;
    auto q = rand_vec(DIM, 999999);

    auto t0 = Clock::now();
    for (int i = 0; i < Q; ++i) {
        auto r = inmem.search(q.data(), K);
        (void)r;
    }
    auto t1 = Clock::now();
    for (int i = 0; i < Q; ++i) {
        auto r = paged.search(q.data(), K);
        (void)r;
    }
    auto t2 = Clock::now();

    double inmem_us = std::chrono::duration<double,std::micro>(t1-t0).count() / Q;
    double paged_us = std::chrono::duration<double,std::micro>(t2-t1).count() / Q;
    std::cout << "[PagedIndex Benchmark] N=" << N << " dim=" << DIM << " k=" << K << "\n"
              << "  In-memory search: " << inmem_us << " us/query\n"
              << "  Paged search:     " << paged_us << " us/query\n"
              << "  Overhead:         " << paged_us / inmem_us << "x\n";
    SUCCEED();

    remove(data_path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
