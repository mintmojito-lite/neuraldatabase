#include "paged_index.hpp"
#include "distance.hpp"
#include <stdexcept>
#include <cstring>
#include <cstdio>

namespace neuraldb {

// ── PagedVectorStore ──────────────────────────────────────────────────────────

PagedVectorStore::PagedVectorStore(const std::string& path, size_t dim,
                                   size_t cache_bytes)
    : path_(path), dim_(dim)
{
    // floats per page (excluding 8-byte header)
    size_t usable = PAGED_PAGE_SIZE - sizeof(uint64_t); // 4088 bytes
    vectors_per_page_ = usable / (dim * sizeof(float));
    if (vectors_per_page_ == 0)
        throw std::invalid_argument("PagedVectorStore: dim too large for 4KB pages");

    // How many pages to keep in cache
    size_t bytes_per_page = PAGED_PAGE_SIZE;
    max_cached_pages_ = std::max(size_t(1), cache_bytes / bytes_per_page);

#ifdef _WIN32
    hFile_ = CreateFileA(path.c_str(),
                         GENERIC_READ | GENERIC_WRITE,
                         FILE_SHARE_READ,
                         nullptr,
                         OPEN_ALWAYS,        // create if absent
                         FILE_ATTRIBUTE_NORMAL,
                         nullptr);
    if (hFile_ == INVALID_HANDLE_VALUE)
        throw std::runtime_error("PagedVectorStore: cannot open " + path);
    // Determine current vector count from file size
    LARGE_INTEGER fsz; fsz.QuadPart = 0;
    GetFileSizeEx(hFile_, &fsz);
    size_t pages_on_disk = static_cast<size_t>(fsz.QuadPart) / PAGED_PAGE_SIZE;
    n_vectors_ = pages_on_disk * vectors_per_page_;
#else
    fd_ = open(path.c_str(), O_RDWR | O_CREAT, 0644);
    if (fd_ < 0)
        throw std::runtime_error("PagedVectorStore: cannot open " + path);
    struct stat sb; fstat(fd_, &sb);
    size_t pages_on_disk = static_cast<size_t>(sb.st_size) / PAGED_PAGE_SIZE;
    n_vectors_ = pages_on_disk * vectors_per_page_;
#endif
}

PagedVectorStore::~PagedVectorStore() {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    for (auto& [pid, cp] : page_cache_) {
#ifdef _WIN32
        if (cp.mapping) UnmapViewOfFile(cp.mapping);
        if (cp.hPageMap) CloseHandle(cp.hPageMap);
#else
        if (cp.mapping && cp.mapping != MAP_FAILED)
            munmap(cp.mapping, cp.mapped_len);
#endif
    }
#ifdef _WIN32
    if (hFile_ != INVALID_HANDLE_VALUE) CloseHandle(hFile_);
#else
    if (fd_ >= 0) close(fd_);
#endif
}

uint32_t PagedVectorStore::push(const float* vec) {
    uint32_t slot = static_cast<uint32_t>(n_vectors_++);
    uint32_t page_id  = static_cast<uint32_t>(slot / vectors_per_page_);
    uint32_t slot_in_page = static_cast<uint32_t>(slot % vectors_per_page_);
    uint64_t page_offset  = static_cast<uint64_t>(page_id) * PAGED_PAGE_SIZE;

    // Allocate / extend file if needed
    size_t vec_offset_in_page = sizeof(uint64_t) + slot_in_page * dim_ * sizeof(float);
    uint64_t file_write_offset = page_offset + vec_offset_in_page;

#ifdef _WIN32
    LARGE_INTEGER li; li.QuadPart = static_cast<LONGLONG>(file_write_offset);
    SetFilePointerEx(hFile_, li, nullptr, FILE_BEGIN);
    DWORD written = 0;
    WriteFile(hFile_, vec, static_cast<DWORD>(dim_ * sizeof(float)), &written, nullptr);
    // Also write page header (page_id) if this is the first slot in the page
    if (slot_in_page == 0) {
        LARGE_INTEGER hdr_li; hdr_li.QuadPart = static_cast<LONGLONG>(page_offset);
        SetFilePointerEx(hFile_, hdr_li, nullptr, FILE_BEGIN);
        DWORD hw = 0;
        WriteFile(hFile_, &page_id, sizeof(uint32_t), &hw, nullptr);
    }
    FlushFileBuffers(hFile_);
#else
    // Extend file with pwrite
    pwrite(fd_, vec, dim_ * sizeof(float), static_cast<off_t>(file_write_offset));
    if (slot_in_page == 0) {
        pwrite(fd_, &page_id, sizeof(uint32_t), static_cast<off_t>(page_offset));
    }
    fsync(fd_);
#endif

    // Invalidate cached page so next read sees fresh data
    {
        std::lock_guard<std::mutex> lk(cache_mutex_);
        auto it = page_cache_.find(page_id);
        if (it != page_cache_.end()) {
#ifdef _WIN32
            if (it->second.mapping) UnmapViewOfFile(it->second.mapping);
            if (it->second.hPageMap) CloseHandle(it->second.hPageMap);
#else
            if (it->second.mapping && it->second.mapping != MAP_FAILED)
                munmap(it->second.mapping, it->second.mapped_len);
#endif
            lru_order_.erase(it->second.lru_it);
            page_cache_.erase(it);
        }
    }
    return slot;
}

void PagedVectorStore::load_page_(uint32_t page_id) const {
    // Must be called with cache_mutex_ held

    if (page_cache_.size() >= max_cached_pages_)
        evict_lru_();

    CachedPage cp;
    cp.page_id = page_id;
    uint64_t page_offset = static_cast<uint64_t>(page_id) * PAGED_PAGE_SIZE;

#ifdef _WIN32
    LARGE_INTEGER file_size;
    GetFileSizeEx(hFile_, &file_size);
    if (page_offset + PAGED_PAGE_SIZE > static_cast<uint64_t>(file_size.QuadPart)) {
        // Page doesn't exist yet — return a zero buffer
        cp.mapping = calloc(1, PAGED_PAGE_SIZE);
        cp.hPageMap = nullptr;
    } else {
        // Map the entire file and offset into the view
        cp.hPageMap = CreateFileMappingA(hFile_, nullptr, PAGE_READWRITE, 0, 0, nullptr);
        if (!cp.hPageMap) throw std::runtime_error("PagedVectorStore: CreateFileMapping failed");
        // MapViewOfFile requires offset aligned to allocation granularity (64KB on Windows)
        // We map the whole file and compute a pointer offset into it
        void* base = MapViewOfFile(cp.hPageMap, FILE_MAP_READ, 0, 0, 0);
        if (!base) {
            CloseHandle(cp.hPageMap);
            throw std::runtime_error("PagedVectorStore: MapViewOfFile failed");
        }
        cp.mapping = static_cast<char*>(base) + page_offset;
    }
#else
    size_t map_len = PAGED_PAGE_SIZE;
    struct stat sb; fstat(fd_, &sb);
    if (page_offset + PAGED_PAGE_SIZE > static_cast<uint64_t>(sb.st_size)) {
        cp.mapping = calloc(1, PAGED_PAGE_SIZE);
        cp.mapped_len = 0; // signal it's a heap alloc not mmap
    } else {
        void* m = mmap(nullptr, map_len, PROT_READ, MAP_PRIVATE, fd_,
                       static_cast<off_t>(page_offset));
        if (m == MAP_FAILED) throw std::runtime_error("PagedVectorStore: mmap failed");
        cp.mapping     = m;
        cp.mapped_len  = map_len;
        cp.mapped_offset = page_offset;
    }
#endif

    lru_order_.push_front(page_id);
    cp.lru_it = lru_order_.begin();
    page_cache_.emplace(page_id, std::move(cp));
}

void PagedVectorStore::evict_lru_() const {
    if (lru_order_.empty()) return;
    uint32_t evict_id = lru_order_.back();
    lru_order_.pop_back();
    auto it = page_cache_.find(evict_id);
    if (it != page_cache_.end()) {
#ifdef _WIN32
        if (it->second.hPageMap) {
            // The view base was mapped over the whole file; we stored a pointer offset.
            // Unmap from the original base (page_offset was added so we subtract it back)
            uint64_t off = static_cast<uint64_t>(evict_id) * PAGED_PAGE_SIZE;
            void* base = static_cast<char*>(it->second.mapping) - off;
            UnmapViewOfFile(base);
            CloseHandle(it->second.hPageMap);
        } else {
            free(it->second.mapping); // heap-alloc stub
        }
#else
        if (it->second.mapped_len > 0)
            munmap(it->second.mapping, it->second.mapped_len);
        else
            free(it->second.mapping);
#endif
        page_cache_.erase(it);
    }
}

const float* PagedVectorStore::get(uint32_t slot) const {
    uint32_t page_id      = static_cast<uint32_t>(slot / vectors_per_page_);
    uint32_t slot_in_page = static_cast<uint32_t>(slot % vectors_per_page_);

    std::lock_guard<std::mutex> lk(cache_mutex_);
    auto it = page_cache_.find(page_id);
    if (it == page_cache_.end()) {
        load_page_(page_id);
        it = page_cache_.find(page_id);
    } else {
        // Move to front (most recently used)
        lru_order_.erase(it->second.lru_it);
        lru_order_.push_front(page_id);
        it->second.lru_it = lru_order_.begin();
    }

    const char* page_base = static_cast<const char*>(it->second.mapping);
    // Skip the 8-byte header (uint64_t page_id stored there during write)
    const float* vec_base = reinterpret_cast<const float*>(page_base + sizeof(uint64_t));
    return vec_base + slot_in_page * dim_;
}

// ── PagedIndex ────────────────────────────────────────────────────────────────

PagedIndex::PagedIndex(const std::string& data_path, size_t dim,
                       size_t cache_bytes, HNSWConfig cfg)
    : dim_(dim),
      store_(data_path, dim, cache_bytes),
      hnsw_(dim, cfg)
{
    // Install a vector accessor so HNSWIndex reads from the paged store
    hnsw_.set_vector_accessor([this](uint32_t internal_id) -> const float* {
        return store_.get(internal_id);
    });
}

void PagedIndex::insert(uint64_t id, const float* vec, std::string metadata) {
    store_.push(vec);
    hnsw_.insert(id, vec, std::move(metadata)); // also stores in hnsw raw_vectors_
}

void PagedIndex::insert(uint64_t id, const std::vector<float>& vec, std::string metadata) {
    if (vec.size() != dim_) throw std::invalid_argument("dimension mismatch");
    insert(id, vec.data(), std::move(metadata));
}

std::vector<std::pair<float, uint64_t>> PagedIndex::search(const float* query, int k) const {
    return hnsw_.search(query, k);
}

std::vector<std::pair<float, uint64_t>> PagedIndex::search(const std::vector<float>& query, int k) const {
    if (query.size() != dim_) throw std::invalid_argument("dimension mismatch");
    return hnsw_.search(query.data(), k);
}

} // namespace neuraldb
