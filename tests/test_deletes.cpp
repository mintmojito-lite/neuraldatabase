/// test_deletes.cpp
/// Task 1: Dynamic deletes (soft deletion) — comprehensive test suite.
#include <gtest/gtest.h>
#include "hnsw.hpp"
#include "vector_store.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <unordered_set>

using namespace neuraldb;

static std::vector<float> rand_vec(size_t dim, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    float norm = 0.f;
    for (auto x : v) norm += x * x;
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
}

// ── FlatIndex delete tests ────────────────────────────────────────────────────

TEST(FlatDelete, DeletedVectorAbsentFromSearch) {
    FlatIndex idx(64);
    for (int i = 0; i < 1000; ++i)
        idx.insert(i, rand_vec(64, i), "{}");

    // Delete every even ID
    for (int i = 0; i < 1000; i += 2)
        idx.delete_vector(i);

    auto q = rand_vec(64, 99999);
    auto results = idx.search(q, 50);
    for (auto& r : results)
        EXPECT_NE(r.id % 2, 0u) << "Deleted vector id=" << r.id << " appeared in results";
}

TEST(FlatDelete, CompactThenSearch) {
    FlatIndex idx(64);
    for (int i = 0; i < 500; ++i)
        idx.insert(i, rand_vec(64, i), "{}");

    for (int i = 0; i < 500; i += 2)
        idx.delete_vector(i);

    idx.compact(); // physically removes deleted entries
    EXPECT_EQ(idx.size(), 250u);

    auto q = rand_vec(64, 99999);
    auto results = idx.search(q, 20);
    for (auto& r : results)
        EXPECT_NE(r.id % 2, 0u) << "Deleted id=" << r.id << " survived compact";
}

// ── HNSWIndex delete tests ─────────────────────────────────────────────────────

TEST(HNSWDelete, DeletedVectorAbsentFrom10kSearch) {
    const int N = 10000, DIM = 128, K = 20;
    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 200;
    HNSWIndex hnsw(DIM, cfg);

    for (int i = 0; i < N; ++i)
        hnsw.insert(i, rand_vec(DIM, i + 1000), "{}");

    // Delete 1000 random vectors (IDs 0..999)
    std::unordered_set<uint64_t> deleted;
    for (int i = 0; i < 1000; ++i) {
        hnsw.delete_vector(i);
        deleted.insert(i);
    }

    // Check across 50 queries
    for (int qi = 0; qi < 50; ++qi) {
        auto q = rand_vec(DIM, qi + 77777);
        auto results = hnsw.search(q, K);
        for (auto& [score, id] : results) {
            EXPECT_EQ(deleted.count(id), 0u)
                << "Deleted vector id=" << id << " appeared in search results";
        }
    }
}

TEST(HNSWDelete, RecallUnchangedOnLiveVectors) {
    const int N = 5000, DIM = 128, K = 10, Q = 50;
    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 200;
    HNSWIndex hnsw(DIM, cfg);
    FlatIndex flat(DIM);

    for (int i = 0; i < N; ++i) {
        auto v = rand_vec(DIM, i + 54321);
        hnsw.insert(i, v, "{}");
        flat.insert(i, v, "{}");
    }

    // Delete first 500
    std::unordered_set<uint64_t> deleted;
    for (int i = 0; i < 500; ++i) {
        hnsw.delete_vector(i);
        flat.delete_vector(i);
        deleted.insert(i);
    }

    float total_recall = 0.f;
    for (int qi = 0; qi < Q; ++qi) {
        auto q = rand_vec(DIM, qi + 88888);

        auto flat_res = flat.search(q, K);
        std::vector<uint64_t> true_ids;
        for (auto& r : flat_res) true_ids.push_back(r.id);

        // Verify flat ground truth has no deleted IDs
        for (auto id : true_ids)
            EXPECT_EQ(deleted.count(id), 0u);

        auto hnsw_res = hnsw.search(q, K);
        for (auto& [score, id] : hnsw_res)
            EXPECT_EQ(deleted.count(id), 0u);

        int found = 0;
        for (auto& [score, id] : hnsw_res)
            if (std::find(true_ids.begin(), true_ids.end(), id) != true_ids.end())
                ++found;
        total_recall += static_cast<float>(found) / K;
    }
    float recall = total_recall / Q;
    std::cout << "[HNSWDelete Recall@10 after 10% delete]: " << recall << "\n";
    EXPECT_GE(recall, 0.85f) << "Recall dropped too much after deletes";
}

TEST(HNSWDelete, CompactSaveLoad) {
    const int DIM = 64;
    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 100; cfg.ef_search = 100;
    HNSWIndex hnsw(DIM, cfg);
    for (int i = 0; i < 500; ++i)
        hnsw.insert(i, rand_vec(DIM, i), "{}");

    // Delete 100 vectors
    std::unordered_set<uint64_t> deleted;
    for (int i = 0; i < 100; ++i) {
        hnsw.delete_vector(i);
        deleted.insert(i);
    }

    const std::string idx_path = "test_compact_tmp.ndb";
    const std::string wal_path = "test_compact_tmp.wal";

    // compact() rebuilds without deleted vectors
    hnsw.compact(idx_path, wal_path);

    // Load compacted index
    HNSWIndex loaded(DIM, cfg);
    loaded.load(idx_path);
    EXPECT_EQ(loaded.live_size(), 400u);

    // Verify no deleted IDs appear in search
    auto q = rand_vec(DIM, 9999);
    auto results = loaded.search(q, 20);
    for (auto& [score, id] : results)
        EXPECT_EQ(deleted.count(id), 0u) << "Deleted id=" << id << " in compacted index";

    remove(idx_path.c_str());
    remove(wal_path.c_str());
}

TEST(HNSWDelete, IsDeletedQuery) {
    HNSWIndex hnsw(32);
    for (int i = 0; i < 10; ++i)
        hnsw.insert(i, rand_vec(32, i), "{}");
    hnsw.delete_vector(3);
    hnsw.delete_vector(7);
    EXPECT_TRUE(hnsw.is_deleted(3));
    EXPECT_TRUE(hnsw.is_deleted(7));
    EXPECT_FALSE(hnsw.is_deleted(0));
    EXPECT_FALSE(hnsw.is_deleted(5));
    EXPECT_EQ(hnsw.live_size(), 8u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
