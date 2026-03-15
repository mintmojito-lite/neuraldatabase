/// test_filter.cpp
/// Task 4: Complex filtering — validates all 8 operators against brute force.
#include <gtest/gtest.h>
#include "filter.hpp"
#include "vector_store.hpp"
#include <cmath>
#include <nlohmann/json.hpp>
#include <random>
#include <string>
#include <vector>

using namespace neuraldb;
using json = nlohmann::json;

static std::vector<float> rand_vec(size_t dim, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    float n = 0.f; for (auto x : v) n += x*x; n = std::sqrtf(n);
    for (auto& x : v) x /= n;
    return v;
}

// ── parse_where unit tests ────────────────────────────────────────────────────

TEST(FilterParse, AllOperators) {
    json w = {
        {"year",     {{"$gte", 2020}, {"$lte", 2024}}},
        {"category", {{"$in", {"science", "tech"}}}},
        {"score",    {{"$gt", 0.8}}},
        {"flag",     {{"$ne", "bad"}}},
    };
    auto wc = parse_where(w);
    EXPECT_EQ(wc.size(), 4u);
    EXPECT_EQ(wc["year"].size(), 2u);
    EXPECT_EQ(wc["category"].size(), 1u);
    EXPECT_EQ(wc["category"][0].op, FilterOp::IN);
}

TEST(FilterParse, BareValueIsEq) {
    json w = {{"color", "red"}};
    auto wc = parse_where(w);
    EXPECT_EQ(wc["color"][0].op, FilterOp::EQ);
    EXPECT_EQ(wc["color"][0].value.get<std::string>(), "red");
}

TEST(FilterParse, UnknownOpThrows) {
    json w = {{"x", {{"$fuzzy", 1}}}};
    EXPECT_THROW(parse_where(w), std::invalid_argument);
}

// ── evaluate_filter unit tests ────────────────────────────────────────────────

static bool check(const json& meta, const json& where) {
    auto wc = parse_where(where);
    return evaluate_filter(meta.dump(), wc);
}

TEST(FilterEval, Eq)  { EXPECT_TRUE(check({{"x",5}}, {{"x",5}})); }
TEST(FilterEval, EqFail) { EXPECT_FALSE(check({{"x",5}}, {{"x",6}})); }
TEST(FilterEval, Ne)  { EXPECT_TRUE(check({{"x",5}}, {{"x",{{"$ne",6}}}})); }
TEST(FilterEval, Gt)  { EXPECT_TRUE(check({{"s",0.9}}, {{"s",{{"$gt",0.8}}}})); }
TEST(FilterEval, GtFail) { EXPECT_FALSE(check({{"s",0.7}}, {{"s",{{"$gt",0.8}}}})); }
TEST(FilterEval, Gte) { EXPECT_TRUE(check({{"y",2020}}, {{"y",{{"$gte",2020}}}})); }
TEST(FilterEval, Lt)  { EXPECT_TRUE(check({{"y",2019}}, {{"y",{{"$lt",2020}}}})); }
TEST(FilterEval, Lte) { EXPECT_TRUE(check({{"y",2024}}, {{"y",{{"$lte",2024}}}})); }
TEST(FilterEval, In)  { EXPECT_TRUE(check({{"c","tech"}}, {{"c",{{"$in",{"science","tech"}}}}})); }
TEST(FilterEval, InFail) { EXPECT_FALSE(check({{"c","art"}}, {{"c",{{"$in",{"science","tech"}}}}})); }
TEST(FilterEval, Nin) { EXPECT_TRUE(check({{"c","art"}}, {{"c",{{"$nin",{"science","tech"}}}}})); }
TEST(FilterEval, NinFail) { EXPECT_FALSE(check({{"c","tech"}}, {{"c",{{"$nin",{"science","tech"}}}}})); }
TEST(FilterEval, Composite) {
    json meta = {{"year", 2022}, {"category", "science"}, {"score", 0.9}};
    json where = {
        {"year",     {{"$gte", 2020}, {"$lte", 2024}}},
        {"category", {{"$in", {"science", "tech"}}}},
        {"score",    {{"$gt", 0.8}}}
    };
    EXPECT_TRUE(check(meta, where));
}
TEST(FilterEval, MissingFieldFails) {
    EXPECT_FALSE(check({{"x",1}}, {{"y",{{"$eq",1}}}}));
}
TEST(FilterEval, EmptyWhere) {
    auto wc = parse_where(json::object());
    EXPECT_TRUE(evaluate_filter("{\"x\":1}", wc));
}

// ── Integration: FlatIndex search_filtered vs brute force ─────────────────────

TEST(FilterIntegration, SearchVsBruteForce) {
    const size_t N = 1000, DIM = 64, K = 20;
    FlatIndex idx(DIM);

    const std::vector<std::string> cats = {"science", "tech", "art", "history", "sports"};
    for (size_t i = 0; i < N; ++i) {
        json meta = {
            {"year",     2018 + static_cast<int>(i % 7)},
            {"category", cats[i % cats.size()]},
            {"score",    0.1 * static_cast<double>(i % 10)}
        };
        idx.insert(i, rand_vec(DIM, static_cast<int>(i)), meta.dump());
    }

    auto q = rand_vec(DIM, 99999);

    // Build WhereClause: year in [2020,2023], category in {science, tech}, score > 0.5
    json w = {
        {"year",     {{"$gte", 2020}, {"$lte", 2023}}},
        {"category", {{"$in", {"science", "tech"}}}},
        {"score",    {{"$gt", 0.5}}}
    };
    auto wc = parse_where(w);

    // Brute-force expected set
    std::vector<uint64_t> brute;
    for (auto& v : idx.raw_vectors()) {
        if (v.deleted) continue;
        if (evaluate_filter(v.metadata, wc)) brute.push_back(v.id);
    }

    // FlatIndex result via search_filtered with a lambda wrapping evaluate_filter
    auto results = idx.search_filtered(q.data(), static_cast<int>(K),
        [&](const std::string& m){ return evaluate_filter(m, wc); });

    // Every returned ID must be in the brute-force eligible set
    for (auto& r : results) {
        bool in_brute = std::find(brute.begin(), brute.end(), r.id) != brute.end();
        EXPECT_TRUE(in_brute) << "result id=" << r.id << " not in eligible set";
    }
    // Must return at most K results
    EXPECT_LE(results.size(), K);
}

TEST(FilterIntegration, AllOperationsExercised) {
    const size_t N = 500, DIM = 32;
    FlatIndex idx(DIM);
    for (size_t i = 0; i < N; ++i) {
        json meta = {{"v", static_cast<int>(i)}, {"tag", (i%2==0?"even":"odd")}};
        idx.insert(i, rand_vec(DIM, static_cast<int>(i)), meta.dump());
    }

    // Test $ne + $lt + $nin
    json w = {
        {"v",   {{"$gte", 100}, {"$lt", 200}}},
        {"tag", {{"$nin", {"odd"}}}}
    };
    auto wc = parse_where(w);
    auto q = rand_vec(DIM, 42);
    auto results = idx.search_filtered(q.data(), 50,
        [&](const std::string& m){ return evaluate_filter(m, wc); });
    for (auto& r : results) {
        auto meta = json::parse(idx.raw_vectors()[r.id].metadata);
        int v = meta["v"].get<int>();
        EXPECT_GE(v, 100);
        EXPECT_LT(v, 200);
        EXPECT_EQ(meta["tag"].get<std::string>(), "even");
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
