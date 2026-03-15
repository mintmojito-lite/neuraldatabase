#pragma once
#include <cstdint>
#include <string>

namespace neuraldb {

/// Memory-mapped file binary header.
struct FileHeader {
    uint32_t magic       = 0x4E444220; // "NDB "
    uint32_t version     = 1;
    uint32_t dim         = 0;
    uint32_t reserved    = 0;
    uint64_t num_vectors = 0;
};

/// RAII wrapper around Windows MapViewOfFile / Linux mmap.
class MMapReader {
public:
    explicit MMapReader(const std::string& path);
    ~MMapReader();

    const void* data() const noexcept { return data_; }
    size_t      size() const noexcept { return size_; }

private:
    void*   data_ = nullptr;
    size_t  size_ = 0;
#ifdef _WIN32
    void*   hFile_    = nullptr;
    void*   hMapping_ = nullptr;
#else
    int     fd_ = -1;
#endif
};

/// Sequential write helper (buffered, no mmap needed for writes).
class FileWriter {
public:
    explicit FileWriter(const std::string& path);
    ~FileWriter();

    template<typename T>
    void write(const T& val) { write_raw(&val, sizeof(T)); }
    void write_raw(const void* data, size_t bytes);
    void flush();

private:
    FILE* fp_ = nullptr;
};

} // namespace neuraldb
