#include "storage.hpp"
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace neuraldb {

// ── MMapReader ────────────────────────────────────────────────────────────────
MMapReader::MMapReader(const std::string& path) {
#ifdef _WIN32
    hFile_ = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                         OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (hFile_ == INVALID_HANDLE_VALUE)
        throw std::runtime_error("MMapReader: cannot open " + path);

    LARGE_INTEGER fsize;
    GetFileSizeEx(static_cast<HANDLE>(hFile_), &fsize);
    size_ = static_cast<size_t>(fsize.QuadPart);

    hMapping_ = CreateFileMappingA(static_cast<HANDLE>(hFile_), nullptr,
                                   PAGE_READONLY, 0, 0, nullptr);
    if (!hMapping_) {
        CloseHandle(static_cast<HANDLE>(hFile_));
        throw std::runtime_error("MMapReader: CreateFileMapping failed");
    }

    data_ = MapViewOfFile(static_cast<HANDLE>(hMapping_), FILE_MAP_READ, 0, 0, 0);
    if (!data_) {
        CloseHandle(static_cast<HANDLE>(hMapping_));
        CloseHandle(static_cast<HANDLE>(hFile_));
        throw std::runtime_error("MMapReader: MapViewOfFile failed");
    }
#else
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("MMapReader: cannot open " + path);

    struct stat sb;
    fstat(fd_, &sb);
    size_ = static_cast<size_t>(sb.st_size);

    data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("MMapReader: mmap failed");
    }
#endif
}

MMapReader::~MMapReader() {
#ifdef _WIN32
    if (data_)     UnmapViewOfFile(data_);
    if (hMapping_) CloseHandle(static_cast<HANDLE>(hMapping_));
    if (hFile_)    CloseHandle(static_cast<HANDLE>(hFile_));
#else
    if (data_ && data_ != (void*)-1) munmap(data_, size_);
    if (fd_ >= 0) close(fd_);
#endif
}

// ── FileWriter ────────────────────────────────────────────────────────────────
FileWriter::FileWriter(const std::string& path) {
    fp_ = fopen(path.c_str(), "wb");
    if (!fp_) throw std::runtime_error("FileWriter: cannot open " + path);
}

FileWriter::~FileWriter() {
    if (fp_) { fflush(fp_); fclose(fp_); }
}

void FileWriter::write_raw(const void* data, size_t bytes) {
    if (fwrite(data, 1, bytes, fp_) != bytes)
        throw std::runtime_error("FileWriter: write error");
}

void FileWriter::flush() {
    if (fp_) fflush(fp_);
}

} // namespace neuraldb
