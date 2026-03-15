#pragma once
/// PagedIndex — a disk-based HNSW index with an LRU page cache.
///
/// Vector data is stored in fixed 4096-byte pages on disk.
/// Only the LRU-N most recently accessed pages are kept in RAM.
/// The HNSW graph itself is kept fully in RAM (it is much smaller).
///
/// Architecture:
///   - PagedVectorStore   : manages the disk file & LRU cache
///   - PagedIndex         : wraps PagedVectorStore + HNSWIndex with
///                          a custom vector accessor hook
#include "hnsw.hpp"
#include <cstdint>
#include <cstddef>
#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace neuraldb {

static constexpr size_t PAGED_PAGE_SIZE = 4096; // 4 KiB

/// One entry in the LRU cache
struct CachedPage {
    void*    mapping  = nullptr; // memory-mapped view of this page
    uint32_t page_id  = 0;
    // iterator into lru_order_ for O(1) eviction
    std::list<uint32_t>::iterator lru_it;

#ifdef _WIN32
    HANDLE hPageMap = nullptr;
#else
    int fd = -1;
    size_t mapped_offset = 0;
    size_t mapped_len    = 0;
#endif
};

/// Manages raw vector data stored in a flat file as fixed 4 KiB pages.
class PagedVectorStore {
public:
    /// @param path        Path to the on-disk data file.
    /// @param dim         Vector dimension (floats per vector).
    /// @param cache_bytes Maximum bytes to hold in RAM (default 1 GiB).
    PagedVectorStore(const std::string& path, size_t dim,
                     size_t cache_bytes = (size_t)1 << 30);
    ~PagedVectorStore();

    /// Append a vector and return its slot index (0-based).
    uint32_t push(const float* vec);

    /// Retrieve a pointer to vector[slot]. May load & cache the page.
    const float* get(uint32_t slot) const;

    size_t count() const noexcept { return n_vectors_; }
    size_t dim()   const noexcept { return dim_; }

private:
    void load_page_(uint32_t page_id) const;
    void evict_lru_() const;

    std::string path_;
    size_t dim_;
    size_t vectors_per_page_;
    size_t max_cached_pages_;
    size_t n_vectors_ = 0;

    mutable std::mutex            cache_mutex_;
    mutable std::list<uint32_t>   lru_order_;
    mutable std::unordered_map<uint32_t, CachedPage> page_cache_;

    // File handle for appending & page-mapping
#ifdef _WIN32
    HANDLE hFile_ = INVALID_HANDLE_VALUE;
#else
    int    fd_    = -1;
#endif
};

/// HNSW index whose vector data lives on disk, paged on demand.
class PagedIndex {
public:
    /// @param data_path  Path to the vector page file (created if absent).
    /// @param dim        Vector dimension.
    /// @param cache_bytes LRU cache budget in bytes (default 1 GiB).
    /// @param cfg        HNSW configuration.
    PagedIndex(const std::string& data_path, size_t dim,
               size_t cache_bytes = (size_t)1 << 30,
               HNSWConfig cfg = {});

    void insert(uint64_t id, const float* vec, std::string metadata = "");
    void insert(uint64_t id, const std::vector<float>& vec, std::string metadata = "");

    std::vector<std::pair<float, uint64_t>> search(const float* query, int k) const;
    std::vector<std::pair<float, uint64_t>> search(const std::vector<float>& query, int k) const;

    size_t size() const { return store_.count(); }
    size_t dim()  const { return dim_; }

private:
    size_t             dim_;
    PagedVectorStore   store_;
    HNSWIndex          hnsw_;
};

} // namespace neuraldb
