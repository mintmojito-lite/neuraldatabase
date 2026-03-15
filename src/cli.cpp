#include "vector_store.hpp"
#include "hnsw.hpp"
#include "distance.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <vector>
#include <random>
#include <string>
#include <numeric>
#include <iostream>
#include <fstream>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <psapi.h>
static size_t get_memory_bytes() {
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    return pmc.WorkingSetSize;
}
#else
#include <sys/resource.h>
static size_t get_memory_bytes() {
    struct rusage r; getrusage(RUSAGE_SELF, &r);
    return static_cast<size_t>(r.ru_maxrss) * 1024;
}
#endif

using namespace neuraldb;
using Clock = std::chrono::high_resolution_clock;

static std::vector<float> rand_vec(size_t dim, int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    float n = 0.f; for (auto x : v) n += x*x; n = std::sqrt(n);
    for (auto& x : v) x /= n;
    return v;
}

static float compute_recall(
    const std::vector<uint64_t>& truth,
    const std::vector<std::pair<float,uint64_t>>& results, int k)
{
    int found = 0;
    for (auto& r : results)
        if (std::find(truth.begin(), truth.end(), r.second) != truth.end())
            ++found;
    return static_cast<float>(found) / static_cast<float>(k);
}

static std::vector<float> load_npy(const std::string& path, size_t& n, size_t& d) {
    std::ifstream fs(path, std::ios::binary);
    if (!fs) throw std::runtime_error("Cannot open " + path);
    fs.seekg(0, std::ios::end);
    size_t file_size = fs.tellg();
    fs.seekg(0, std::ios::beg);
    char prologue[10]; fs.read(prologue, 10);
    uint16_t h_len; std::memcpy(&h_len, prologue + 8, 2);
    size_t data_offset = 10 + h_len;
    fs.seekg(data_offset, std::ios::beg);
    
    size_t body_size = file_size - data_offset;
    std::vector<float> data(body_size / sizeof(float));
    fs.read(reinterpret_cast<char*>(data.data()), body_size);
    
    d = (d == 0) ? 384 : d; 
    n = data.size() / d;
    return data;
}

int main(int argc, char* argv[]) {
    std::string emb_path, query_path;
    size_t dim = 0, n = 100000;
    int k = 10;
    bool is_bench = false;

    std::cout << "DEBUG: argc=" << argc << "\n";
    for (int i = 0; i < argc; ++i) {
        std::cout << "DEBUG: argv[" << i << "]=\"" << argv[i] << "\"\n";
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "bench") is_bench = true;
        else if (a.rfind("--embeddings=", 0) == 0) emb_path = a.substr(13);
        else if (a.rfind("--queries=", 0) == 0) query_path = a.substr(10);
        else if (a.rfind("--dim=", 0) == 0) dim = std::stoul(a.substr(6));
        else if (a.rfind("--n=", 0) == 0) n = std::stoul(a.substr(4));
        else if (a.rfind("--k=", 0) == 0) k = std::stoi(a.substr(4));
    }

    std::cout << "DEBUG: is_bench=" << is_bench << " emb_path=\"" << emb_path << "\"\n";

    if (!is_bench) {
        std::cout << "Usage: neuraldb cli bench [--embeddings file.npy] [--queries queries.npy] [--k 10] [--dim 384]\n";
        return 0;
    }

    std::vector<float> train_data, query_data;
    size_t n_train = 0, n_query = 0;

    if (!emb_path.empty()) {
        std::cout << "Loading embeddings from " << emb_path << "...\n";
        train_data = load_npy(emb_path, n_train, dim);
        n = n_train;
    } else {
        if (dim == 0) dim = 768;
        std::cout << "Generating " << n << " random vectors (dim=" << dim << ")...\n";
        train_data.resize(n * dim);
        for (size_t i = 0; i < n; ++i) {
            auto v = rand_vec(dim, (int)i);
            std::memcpy(train_data.data() + i * dim, v.data(), dim * sizeof(float));
        }
    }

    if (!query_path.empty()) {
        std::cout << "Loading queries from " << query_path << "...\n";
        query_data = load_npy(query_path, n_query, dim);
    } else {
        if (dim == 0) dim = 768;
        n_query = 100;
        query_data.resize(n_query * dim);
        for (size_t i = 0; i < n_query; ++i) {
            auto v = rand_vec(dim, (int)i + 999999);
            std::memcpy(query_data.data() + i * dim, v.data(), dim * sizeof(float));
        }
    }

    std::cout << "NeuralDB Benchmark\n";
    std::cout << "  dim=" << dim << "  N=" << n << "  k=" << k << "  queries=" << n_query << "\n\n";

    HNSWConfig cfg; cfg.M = 16; cfg.ef_construction = 200;
    HNSWIndex hnsw(dim, cfg);
    FlatIndex flat(dim);

    std::cout << "Building HNSW index (parallel)..." << std::endl;
    auto t0 = Clock::now();
    #pragma omp parallel for schedule(dynamic, 128)
    for (int i = 0; i < (int)n; ++i)
        hnsw.insert(i, train_data.data() + i * dim, "{}");
    auto t1 = Clock::now();
    double build_s = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "Building Flat index (ground truth)..." << std::endl;
    for (size_t i = 0; i < n; ++i)
        flat.insert(i, train_data.data() + i * dim, "{}");

    std::vector<int> ef_values = { 10, 50, 100, 200, 500 };
    
    std::cout << "| ef_search | Query (ms) | Recall@" << k << " |\n";
    std::cout << "|-----------|------------|-----------|\n";

    for (int ef : ef_values) {
        hnsw.set_ef_search(ef);
        std::vector<double> lats;
        float total_recall = 0;

        for (size_t qi = 0; qi < n_query; ++qi) {
            const float* qptr = query_data.data() + qi * dim;
            auto qa = Clock::now();
            auto res = hnsw.search(qptr, k);
            auto qb = Clock::now();
            lats.push_back(std::chrono::duration<double, std::milli>(qb - qa).count());

            auto truth_res = flat.search(qptr, k);
            std::vector<uint64_t> truth;
            for (auto& r : truth_res) truth.push_back(r.id);
            total_recall += compute_recall(truth, res, k);
        }
        
        std::sort(lats.begin(), lats.end());
        double mean = std::accumulate(lats.begin(), lats.end(), 0.0) / lats.size();
        float recall = total_recall / n_query;

        printf("| %-9d | %-10.3f | %-9.4f |\n", ef, mean, recall);
    }

    double mem_gb = get_memory_bytes() / 1e9;
    std::cout << "\nBuild time: " << build_s << "s\n";
    std::cout << "Memory usage: " << mem_gb << " GB\n";

    return 0;
}
