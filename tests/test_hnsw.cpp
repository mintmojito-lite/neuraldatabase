#include <gtest/gtest.h>
#include "hnsw.hpp"
#include "vector_store.hpp"
#include "distance.hpp"
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>
#include <nlohmann/json.hpp>

using namespace neuraldb;

// ── helpers ────────────────────────────────────────────────────────────────────
static std::vector<float> rand_vec(size_t dim, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    // Normalise
    float norm = 0.f;
    for (auto x : v) norm += x * x;
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
}

// Compute Recall@k: fraction of true top-k found in HNSW top-k
static float compute_recall(
    const std::vector<uint64_t>& true_ids,
    const std::vector<std::pair<float,uint64_t>>& hnsw_results,
    int k)
{
    std::vector<uint64_t> hnsw_ids;
    for (auto& r : hnsw_results) hnsw_ids.push_back(r.second);

    int found = 0;
    for (auto tid : true_ids) {
        if (std::find(hnsw_ids.begin(), hnsw_ids.end(), tid) != hnsw_ids.end())
            ++found;
    }
    return static_cast<float>(found) / static_cast<float>(k);
}

// ── Flat index tests ───────────────────────────────────────────────────────────
TEST(FlatIndex, InsertAndSearch) {
    FlatIndex idx(128);
    auto q = rand_vec(128, 0);
    idx.insert(42, q, R"({"cat":"test"})");

    auto results = idx.search(q, 1);
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].id, 42u);
    EXPECT_NEAR(results[0].score, 1.f, 1e-5f);
}

TEST(FlatIndex, TopKOrdering) {
    FlatIndex idx(128);
    auto q = rand_vec(128, 0);
    for (int i = 1; i <= 100; ++i)
        idx.insert(i, rand_vec(128, i), "{}");

    auto results = idx.search(q, 10);
    ASSERT_EQ(results.size(), 10u);
    // Results should be sorted descending
    for (size_t i = 1; i < results.size(); ++i)
        EXPECT_GE(results[i-1].score, results[i].score);
}

TEST(FlatIndex, MetadataFilter) {
    FlatIndex idx(128);
    auto q = rand_vec(128, 0);
    for (int i = 0; i < 100; ++i) {
        std::string cat = (i % 2 == 0) ? "science" : "art";
        idx.insert(i, rand_vec(128, i), R"({"category":")" + cat + R"("})");
    }

    auto filter = [](const std::string& meta) {
        auto j = nlohmann::json::parse(meta, nullptr, false);
        return !j.is_discarded() && j.value("category", "") == "science";
    };

    auto results = idx.search_filtered(q.data(), 10, filter);
    for (auto& r : results) {
        auto j = nlohmann::json::parse(r.metadata);
        EXPECT_EQ(j["category"], "science");
    }
}

// ── HNSW correctness / Recall@10 ──────────────────────────────────────────────
TEST(HNSWIndex, Recall10) {
    const int N   = 10000;
    const int DIM = 768;
    const int K   = 10;
    const int Q   = 100;

    FlatIndex  flat(DIM);
    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 1000;
    HNSWIndex  hnsw(DIM, cfg);

    for (int i = 0; i < N; ++i) {
        auto v = rand_vec(DIM, i + 99999);
        flat.insert(i, v, "{}");
        hnsw.insert(i, v, "{}");
    }

    float total_recall = 0.f;
    for (int qi = 0; qi < Q; ++qi) {
        auto q = rand_vec(DIM, qi + 77777);

        // Ground truth from flat search
        auto flat_results = flat.search(q, K);
        std::vector<uint64_t> true_ids;
        for (auto& r : flat_results) true_ids.push_back(r.id);

        // HNSW results
        auto hnsw_results = hnsw.search(q, K);
        total_recall += compute_recall(true_ids, hnsw_results, K);
    }

    float recall = total_recall / Q;
    std::cout << "[HNSW Recall@10]: " << recall << std::endl;
    EXPECT_GE(recall, 0.90f) << "Recall@10 below threshold";
}

// ── HNSW save / load round-trip ───────────────────────────────────────────────
TEST(HNSWIndex, SaveLoad) {
    const size_t DIM = 64;
    HNSWIndex hnsw(DIM);
    for (int i = 0; i < 500; ++i)
        hnsw.insert(i, rand_vec(DIM, i), "{}");

    const std::string path = "test_hnsw_tmp.ndb";
    hnsw.save(path);

    HNSWIndex loaded(DIM);
    loaded.load(path);

    EXPECT_EQ(loaded.size(), hnsw.size());

    // Same query should give same result
    auto q = rand_vec(DIM, 9999);
    auto r1 = hnsw.search(q, 5);
    auto r2 = loaded.search(q, 5);
    ASSERT_EQ(r1.size(), r2.size());
    for (size_t i = 0; i < r1.size(); ++i)
        EXPECT_EQ(r1[i].second, r2[i].second);

    remove(path.c_str());
}

// ── WAL recovery test ─────────────────────────────────────────────────────────
#include "wal.hpp"
TEST(WAL, ReplayRecovery) {
    const std::string wal_path = "test_wal_tmp.wal";
    {
        WAL wal(wal_path);
        for (int i = 0; i < 10; ++i) {
            auto v = rand_vec(32, i);
            wal.append_insert(i, v.data(), 32, "{}");
        }
    } // WAL closes

    // Replay
    WAL wal2(wal_path);
    int count = 0;
    wal2.replay([&](const WALRecord& rec){
        EXPECT_EQ(rec.dim, 32u);
        ++count;
    });
    EXPECT_EQ(count, 10);
    wal2.checkpoint();
    remove(wal_path.c_str());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
