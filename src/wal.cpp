#include "wal.hpp"
#include <stdexcept>
#include <cstring>
#include <vector>

namespace neuraldb {

WAL::WAL(const std::string& path) : path_(path) {
    // Open in append+read binary mode
    fp_ = fopen(path.c_str(), "ab");
    if (!fp_) throw std::runtime_error("WAL: cannot open " + path);
}

WAL::~WAL() {
    if (fp_) fclose(fp_);
}

void WAL::append_insert(uint64_t id, const float* vec, size_t dim,
                        const std::string& metadata) {
    uint8_t op = static_cast<uint8_t>(WALOp::INSERT);
    fwrite(&op, 1, 1, fp_);
    fwrite(&id, 8, 1, fp_);
    uint32_t dim32 = static_cast<uint32_t>(dim);
    fwrite(&dim32, 4, 1, fp_);
    fwrite(vec, sizeof(float), dim, fp_);
    uint32_t mlen = static_cast<uint32_t>(metadata.size());
    fwrite(&mlen, 4, 1, fp_);
    fwrite(metadata.data(), 1, mlen, fp_);
    fflush(fp_); // durability: flush after every record
}

void WAL::append_delete(uint64_t id) {
    uint8_t op = static_cast<uint8_t>(WALOp::DELETE);
    fwrite(&op, 1, 1, fp_);
    fwrite(&id, 8, 1, fp_);
    // No vector payload for DELETE
    uint32_t dim32 = 0;
    fwrite(&dim32, 4, 1, fp_);
    uint32_t mlen = 0;
    fwrite(&mlen, 4, 1, fp_);
    fflush(fp_);
}

void WAL::replay(const std::function<void(const WALRecord&)>& on_insert,
                 const std::function<void(uint64_t)>& on_delete) {
    // Re-open in read mode
    FILE* rfp = fopen(path_.c_str(), "rb");
    if (!rfp) return; // no WAL file = no-op

    WALRecord rec;
    while (true) {
        uint8_t op;
        if (fread(&op, 1, 1, rfp) != 1) break;

        uint64_t id;
        if (fread(&id, 8, 1, rfp) != 1) break;
        rec.id = id;
        rec.op = static_cast<WALOp>(op);

        uint32_t dim;
        if (fread(&dim, 4, 1, rfp) != 1) break;
        rec.dim = dim;
        rec.vec.resize(dim);
        if (fread(rec.vec.data(), sizeof(float), dim, rfp) != dim) break;

        uint32_t mlen;
        if (fread(&mlen, 4, 1, rfp) != 1) break;
        rec.metadata.resize(mlen);
        if (mlen > 0 && fread(rec.metadata.data(), 1, mlen, rfp) != mlen) break;

        if (rec.op == WALOp::INSERT) {
            on_insert(rec);
        } else if (rec.op == WALOp::DELETE && on_delete) {
            on_delete(rec.id);
        }
    }
    fclose(rfp);
}

void WAL::checkpoint() {
    if (fp_) { fclose(fp_); fp_ = nullptr; }
    // Truncate by re-opening in write mode
    fp_ = fopen(path_.c_str(), "wb");
    if (!fp_) throw std::runtime_error("WAL: checkpoint failed, cannot truncate " + path_);
}

} // namespace neuraldb
