#pragma once
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace neuraldb {

enum class WALOp : uint8_t { INSERT = 1, DELETE = 2 };

struct WALRecord {
    WALOp       op;
    uint64_t    id;
    uint32_t    dim;
    std::vector<float> vec;
    std::string metadata;
};

/// Write-ahead log for crash recovery.
class WAL {
public:
    explicit WAL(const std::string& path);
    ~WAL();

    void append_insert(uint64_t id, const float* vec, size_t dim,
                       const std::string& metadata);

    /// Write a soft-delete record (no vector payload).
    void append_delete(uint64_t id);

    /// Replay all records.
    /// on_insert is called for INSERT, on_delete for DELETE (may be null).
    void replay(const std::function<void(const WALRecord&)>& on_insert,
                const std::function<void(uint64_t)>& on_delete = nullptr);

    /// Flush + truncate WAL (call after index checkpoint).
    void checkpoint();

    bool is_open() const noexcept { return fp_ != nullptr; }

private:
    std::string path_;
    FILE*       fp_ = nullptr;
};

} // namespace neuraldb
