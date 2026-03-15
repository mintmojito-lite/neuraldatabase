#include <benchmark/benchmark.h>
#include "vector_store.hpp"
#include "hnsw.hpp"
#include "distance.hpp"
#include <random>
#include <vector>

using namespace neuraldb;

// ── helpers ───────────────────────────────────────────────────────────────────
static std::vector<float> rand_vec(size_t dim, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    float n = 0.f;
    for (auto x : v) n += x * x;
    n = std::sqrt(n);
    for (auto& x : v) x /= n;
    return v;
}

// ── distance benchmarks ───────────────────────────────────────────────────────
template<size_t DIM>
static void BM_CosineNaive(benchmark::State& state) {
    auto a = rand_vec(DIM, 1), b = rand_vec(DIM, 2);
    for (auto _ : state)
        benchmark::DoNotOptimize(cosine_similarity_naive(a.data(), b.data(), DIM));
}
BENCHMARK_TEMPLATE(BM_CosineNaive, 128)->Name("Cosine/Naive/128");
BENCHMARK_TEMPLATE(BM_CosineNaive, 256)->Name("Cosine/Naive/256");
BENCHMARK_TEMPLATE(BM_CosineNaive, 768)->Name("Cosine/Naive/768");
BENCHMARK_TEMPLATE(BM_CosineNaive, 1536)->Name("Cosine/Naive/1536");

template<size_t DIM>
static void BM_CosineAVX2(benchmark::State& state) {
    auto a = rand_vec(DIM, 1), b = rand_vec(DIM, 2);
    for (auto _ : state)
        benchmark::DoNotOptimize(cosine_similarity_avx2(a.data(), b.data(), DIM));
}
BENCHMARK_TEMPLATE(BM_CosineAVX2, 128)->Name("Cosine/AVX2/128");
BENCHMARK_TEMPLATE(BM_CosineAVX2, 256)->Name("Cosine/AVX2/256");
BENCHMARK_TEMPLATE(BM_CosineAVX2, 768)->Name("Cosine/AVX2/768");
BENCHMARK_TEMPLATE(BM_CosineAVX2, 1536)->Name("Cosine/AVX2/1536");

template<size_t DIM>
static void BM_L2AVX2(benchmark::State& state) {
    auto a = rand_vec(DIM, 1), b = rand_vec(DIM, 2);
    for (auto _ : state)
        benchmark::DoNotOptimize(l2_distance_avx2(a.data(), b.data(), DIM));
}
BENCHMARK_TEMPLATE(BM_L2AVX2, 128)->Name("L2/AVX2/128");
BENCHMARK_TEMPLATE(BM_L2AVX2, 768)->Name("L2/AVX2/768");

// ── flat index search: 100k vectors @ dim=768, k=10 ──────────────────────────
static FlatIndex* g_flat = nullptr;
static std::vector<float> g_query;

static void SetupFlat(benchmark::State&) {
    const size_t N = 100000, DIM = 768;
    if (!g_flat) {
        g_flat = new FlatIndex(DIM);
        for (size_t i = 0; i < N; ++i)
            g_flat->insert(i, rand_vec(DIM, static_cast<int>(i)), "{}");
        g_query = rand_vec(DIM, 999999);
    }
}

static void BM_FlatSearch_100k(benchmark::State& state) {
    SetupFlat(state);
    for (auto _ : state) {
        auto r = g_flat->search(g_query.data(), 10);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetLabel("N=100k dim=768 k=10");
    state.SetItemsProcessed(state.iterations() * 100000);
}
BENCHMARK(BM_FlatSearch_100k)->Iterations(50);

// ── HNSW search: 100k vectors @ dim=768, k=10 ────────────────────────────────
static HNSWIndex* g_hnsw = nullptr;

static void SetupHNSW(benchmark::State&) {
    const size_t N = 100000, DIM = 768;
    if (!g_hnsw) {
        HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200; cfg.ef_search = 50;
        g_hnsw = new HNSWIndex(DIM, cfg);
        for (size_t i = 0; i < N; ++i)
            g_hnsw->insert(i, rand_vec(DIM, static_cast<int>(i)), "{}");
    }
}

static void BM_HNSWSearch_100k(benchmark::State& state) {
    SetupHNSW(state);
    for (auto _ : state) {
        auto r = g_hnsw->search(g_query.data(), 10);
        benchmark::DoNotOptimize(r.data());
    }
    state.SetLabel("N=100k dim=768 k=10 ef=50");
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_HNSWSearch_100k)->Iterations(500);

// ── HNSW at different ef_search values ───────────────────────────────────────
static void BM_HNSWSearch_ef(benchmark::State& state) {
    SetupHNSW(state);
    g_hnsw->set_ef_search(static_cast<int>(state.range(0)));
    for (auto _ : state) {
        auto r = g_hnsw->search(g_query.data(), 10);
        benchmark::DoNotOptimize(r.data());
    }
}
BENCHMARK(BM_HNSWSearch_ef)->Arg(10)->Arg(50)->Arg(100)->Arg(200)
    ->Iterations(200);

BENCHMARK_MAIN();
