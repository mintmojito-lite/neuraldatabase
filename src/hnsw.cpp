#include "hnsw.hpp"
#include "distance.hpp"
#include "wal.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <vector>
#include <shared_mutex>

namespace neuraldb {

HNSWIndex::HNSWIndex(size_t dim, HNSWConfig cfg)
    : dim_(dim), cfg_(cfg), mL_(1.0f / std::log(static_cast<float>(cfg.M)))
{
    this->num_vectors_.store(0);
}

HNSWIndex::~HNSWIndex() = default;

static thread_local std::mt19937 rng_{std::random_device{}()};

static int random_layer(float mL) {
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float val = dist(rng_);
    if (val < 1e-12f) val = 1e-12f;
    return static_cast<int>(-std::log(val) * mL);
}

void HNSWIndex::insert(uint64_t id, const float* vec, std::string metadata) {
    uint32_t q;
    const int level = random_layer(this->mL_);

    {
        std::unique_lock<std::shared_mutex> lock(this->index_mutex_);
        q = static_cast<uint32_t>(this->num_vectors_.load());
        this->num_vectors_.fetch_add(1);

        this->raw_vectors_.insert(this->raw_vectors_.end(), vec, vec + this->dim_);
        this->id_map_.push_back(id);
        this->metadata_.push_back(std::move(metadata));
        this->tombstone_.push_back(false);   // new vector is alive
        this->graph_.emplace_back(level + 1);

        if (q == 0) {
            this->max_layer_   = level;
            this->entry_point_ = 0;
            return;
        }
    }

    uint32_t ep;
    int cur_max_layer;
    {
        std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
        ep = this->entry_point_;
        cur_max_layer = this->max_layer_;
    }

    for (int lc = cur_max_layer; lc > level; --lc) {
        auto candidates = this->search_layer_(vec, ep, 1, lc);
        if (!candidates.empty()) ep = candidates.front().second;
    }

    for (int lc = std::min(level, cur_max_layer); lc >= 0; --lc) {
        int ef_c = std::max(this->cfg_.ef_construction, this->cfg_.M);
        auto candidates = this->search_layer_(vec, ep, ef_c, lc);
        auto neighbours = this->select_neighbours_(q, candidates, this->cfg_.M);

        {
            std::unique_lock<std::shared_mutex> lock(this->index_mutex_);
            this->graph_[q][lc] = neighbours;
        }

        for (uint32_t nb : neighbours) {
            std::unique_lock<std::shared_mutex> lock(this->index_mutex_);
            if (lc >= static_cast<int>(this->graph_[nb].size()))
                this->graph_[nb].resize(lc + 1);

            auto& nb_links = this->graph_[nb][lc];
            nb_links.push_back(q);

            int Mmax = (lc == 0) ? this->cfg_.M * 2 : this->cfg_.M;
            if (static_cast<int>(nb_links.size()) > Mmax) {
                std::vector<std::pair<float, uint32_t>> cands;
                const float* nb_vec = this->get_vec_(nb);
                for (uint32_t x : nb_links) {
                    cands.push_back({cosine_similarity_avx2(nb_vec, this->get_vec_(x), this->dim_), x});
                }
                nb_links = this->select_neighbours_(nb, cands, Mmax);
            }
        }
        if (!candidates.empty()) ep = candidates.front().second;
    }

    {
        std::unique_lock<std::shared_mutex> lock(this->index_mutex_);
        if (level > this->max_layer_) {
            this->max_layer_   = level;
            this->entry_point_ = q;
        }
    }
}

void HNSWIndex::insert(uint64_t id, const std::vector<float>& vec, std::string metadata) {
    if (vec.size() != this->dim_) throw std::invalid_argument("dimension mismatch");
    this->insert(id, vec.data(), std::move(metadata));
}

// ── Soft delete ───────────────────────────────────────────────────────────────

void HNSWIndex::delete_vector(uint64_t external_id_val) {
    std::unique_lock<std::shared_mutex> lock(this->index_mutex_);
    size_t nv = this->num_vectors_.load();
    for (size_t i = 0; i < nv; ++i) {
        if (this->id_map_[i] == external_id_val) {
            this->tombstone_[i] = true;
            return;
        }
    }
    // If not found, silently ignore (idempotent)
}

bool HNSWIndex::is_deleted(uint64_t external_id_val) const {
    std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
    size_t nv = this->num_vectors_.load();
    for (size_t i = 0; i < nv; ++i) {
        if (this->id_map_[i] == external_id_val)
            return this->tombstone_[i];
    }
    return false;
}

size_t HNSWIndex::live_size() const noexcept {
    std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
    size_t count = 0;
    for (bool t : this->tombstone_) if (!t) ++count;
    return count;
}

void HNSWIndex::compact(const std::string& index_path, const std::string& wal_path) {
    // Build a fresh index from only the live vectors
    HNSWIndex fresh(this->dim_, this->cfg_);
    {
        std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
        size_t nv = this->num_vectors_.load();
        for (size_t i = 0; i < nv; ++i) {
            if (!this->tombstone_[i]) {
                fresh.insert(this->id_map_[i],
                             this->raw_vectors_.data() + i * this->dim_,
                             this->metadata_[i]);
            }
        }
    }
    // Persist compacted index
    fresh.save(index_path);

    // Checkpoint the WAL (truncate it)
    WAL wal(wal_path);
    wal.checkpoint();
}

// ── search_layer_ ─────────────────────────────────────────────────────────────

std::vector<std::pair<float,uint32_t>> HNSWIndex::search_layer_(
    const float* query, uint32_t entry_point_id, int ef, int layer) const
{
    using Pair = std::pair<float, uint32_t>;
    static thread_local std::vector<uint16_t> visited_buf;
    static thread_local uint16_t visit_tag = 0;

    size_t nv = this->num_vectors_.load(std::memory_order_relaxed);
    if (visited_buf.size() < nv) visited_buf.resize(nv + 10000, 0);
    visit_tag++;
    if (visit_tag == 0) { std::fill(visited_buf.begin(), visited_buf.end(), 0); visit_tag = 1; }

    float ep_sim;
    {
        std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
        ep_sim = cosine_similarity_avx2(query, this->get_vec_(entry_point_id), this->dim_);
    }

    std::priority_queue<Pair, std::vector<Pair>, std::greater<Pair>> cands;
    std::priority_queue<Pair> results;

    cands.push({-ep_sim, entry_point_id});
    results.push({-ep_sim, entry_point_id});
    visited_buf[entry_point_id] = visit_tag;

    while (!cands.empty()) {
        auto [neg_c_sim, c] = cands.top();
        cands.pop();
        float c_sim = -neg_c_sim;
        float worst_sim = results.empty() ? -std::numeric_limits<float>::infinity()
                                          : -results.top().first;
        if (c_sim < worst_sim && static_cast<int>(results.size()) >= ef) break;

        std::vector<uint32_t> nbs;
        {
            std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
            if (layer < static_cast<int>(this->graph_[c].size())) nbs = this->graph_[c][layer];
        }

        for (uint32_t nb : nbs) {
            if (visited_buf[nb] == visit_tag) continue;
            visited_buf[nb] = visit_tag;
            float nb_sim;
            {
                std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
                nb_sim = cosine_similarity_avx2(query, this->get_vec_(nb), this->dim_);
            }
            float worst = results.empty() ? -std::numeric_limits<float>::infinity()
                                          : -results.top().first;
            if (nb_sim > worst || static_cast<int>(results.size()) < ef) {
                cands.push({-nb_sim, nb});
                results.push({-nb_sim, nb});
                if (static_cast<int>(results.size()) > ef) results.pop();
            }
        }
    }

    std::vector<Pair> out;
    while (!results.empty()) { out.push_back({-results.top().first, results.top().second}); results.pop(); }
    std::sort(out.begin(), out.end(), [](const Pair& a, const Pair& b){ return a.first > b.first; });
    return out;
}

// ── select_neighbours_ ────────────────────────────────────────────────────────

std::vector<uint32_t> HNSWIndex::select_neighbours_(
    uint32_t /*q*/, const std::vector<std::pair<float,uint32_t>>& candidates, int M) const
{
    std::vector<std::pair<float,uint32_t>> sorted = candidates;
    std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return a.first > b.first; });
    std::vector<uint32_t> result;
    for (int i = 0; i < M && i < static_cast<int>(sorted.size()); ++i) result.push_back(sorted[i].second);
    return result;
}

// ── search ───────────────────────────────────────────────────────────────────

std::vector<std::pair<float, uint64_t>> HNSWIndex::search(const float* query, int k) const {
    size_t nv = this->num_vectors_.load();
    if (nv == 0) return {};

    // Find a valid (live) entry point
    uint32_t ep; int max_l;
    {
        std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
        ep = this->entry_point_;
        max_l = this->max_layer_;
        // If entry point is deleted, find first live vector
        if (this->tombstone_[ep]) {
            bool found = false;
            for (uint32_t i = 0; i < static_cast<uint32_t>(nv); ++i) {
                if (!this->tombstone_[i]) { ep = i; found = true; break; }
            }
            if (!found) return {}; // all deleted
        }
    }

    for (int lc = max_l; lc > 0; --lc) {
        auto candidates = this->search_layer_(query, ep, 1, lc);
        if (!candidates.empty()) ep = candidates.front().second;
    }
    auto candidates = this->search_layer_(query, ep, this->cfg_.ef_search, 0);

    // Collect up to k non-deleted results
    std::vector<std::pair<float, uint64_t>> res_out;
    {
        std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
        for (const auto& cand : candidates) {
            if (!this->tombstone_[cand.second]) {
                res_out.push_back({ cand.first, this->id_map_[cand.second] });
                if (static_cast<int>(res_out.size()) == k) break;
            }
        }
    }
    return res_out;
}

std::vector<std::pair<float, uint64_t>> HNSWIndex::search(const std::vector<float>& query, int k) const {
    if (query.size() != this->dim_) throw std::invalid_argument("dimension mismatch");
    return this->search(query.data(), k);
}

// ── save / load ───────────────────────────────────────────────────────────────

void HNSWIndex::save(const std::string& path) const {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) throw std::runtime_error("Cannot open file for write: " + path);
    auto wwrite = [&](const void* d, size_t n){ fwrite(d, 1, n, fp); };
    uint32_t magic = 0x4E444220, ver = 2, dim32 = static_cast<uint32_t>(this->dim_), res = 0;
    uint64_t nv = this->num_vectors_.load();
    wwrite(&magic, 4); wwrite(&ver, 4); wwrite(&dim32, 4); wwrite(&res, 4); wwrite(&nv, 8);
    {
        std::shared_lock<std::shared_mutex> lock(this->index_mutex_);
        wwrite(this->id_map_.data(), this->id_map_.size() * sizeof(uint64_t));
        wwrite(this->raw_vectors_.data(), this->raw_vectors_.size() * sizeof(float));
        for (size_t i = 0; i < this->metadata_.size(); ++i) {
            uint32_t len = static_cast<uint32_t>(this->metadata_[i].size());
            wwrite(&len, 4);
            wwrite(this->metadata_[i].data(), len);
        }
        wwrite(&this->max_layer_, sizeof(int));
        wwrite(&this->entry_point_, sizeof(uint32_t));
        // Tombstone: pack into uint64 words
        size_t words = (nv + 63) / 64;
        std::vector<uint64_t> tbits(words, 0);
        for (size_t i = 0; i < nv; ++i)
            if (this->tombstone_[i]) tbits[i / 64] |= (uint64_t(1) << (i % 64));
        wwrite(tbits.data(), words * sizeof(uint64_t));
        for (size_t i = 0; i < nv; ++i) {
            uint32_t nlayers = static_cast<uint32_t>(this->graph_[i].size());
            wwrite(&nlayers, 4);
            for (size_t j = 0; j < this->graph_[i].size(); ++j) {
                uint32_t sz = static_cast<uint32_t>(this->graph_[i][j].size());
                wwrite(&sz, 4);
                if (sz) wwrite(this->graph_[i][j].data(), sz * sizeof(uint32_t));
            }
        }
    }
    fclose(fp);
}

void HNSWIndex::load(const std::string& path) {
    FILE* fp = fopen(path.c_str(), "rb");
    if (!fp) throw std::runtime_error("Cannot open file for read: " + path);
    auto rread = [&](void* d, size_t n){ fread(d, 1, n, fp); };
    uint32_t magic, ver, dim32, res; uint64_t nv;
    rread(&magic, 4); rread(&ver, 4); rread(&dim32, 4); rread(&res, 4); rread(&nv, 8);
    if (magic != 0x4E444220) { fclose(fp); throw std::runtime_error("Bad magic"); }
    this->dim_ = dim32;
    this->num_vectors_.store(nv);
    this->id_map_.resize(nv);
    rread(this->id_map_.data(), nv * sizeof(uint64_t));
    this->raw_vectors_.resize(nv * this->dim_);
    rread(this->raw_vectors_.data(), nv * this->dim_ * sizeof(float));
    this->metadata_.resize(nv);
    for (size_t i = 0; i < nv; ++i) {
        uint32_t len; rread(&len, 4);
        this->metadata_[i].resize(len);
        rread(this->metadata_[i].data(), len);
    }
    rread(&this->max_layer_, sizeof(int));
    rread(&this->entry_point_, sizeof(uint32_t));
    // Tombstone
    this->tombstone_.resize(nv, false);
    if (ver >= 2) {
        size_t words = (nv + 63) / 64;
        std::vector<uint64_t> tbits(words, 0);
        rread(tbits.data(), words * sizeof(uint64_t));
        for (size_t i = 0; i < nv; ++i)
            this->tombstone_[i] = (tbits[i / 64] >> (i % 64)) & 1;
    }
    this->graph_.resize(nv);
    for (size_t n = 0; n < nv; ++n) {
        uint32_t nlayers; rread(&nlayers, 4);
        this->graph_[n].resize(nlayers);
        for (size_t l = 0; l < nlayers; ++l) {
            uint32_t sz; rread(&sz, 4);
            this->graph_[n][l].resize(sz);
            if (sz) rread(this->graph_[n][l].data(), sz * sizeof(uint32_t));
        }
    }
    fclose(fp);
}

} // namespace neuraldb
