# NeuralDB 🚀

A high-performance, from-scratch vector database engine written in C++17 with Python bindings.

> **Pure Systems Engineering** — Built without FAISS, Annoy, or external ANN libraries. Features SIMD acceleration, lock-free parallel execution, out-of-RAM paged indexing, and Write-Ahead Logging (WAL) for durability.

---

## 🌟 Key Features

| Feature | Description |
|---|---|
| ⚡ **Hardware Acceleration** | SIMD vector kernels with AVX2 & ARM NEON runtime dispatch for Cosine and L2 metrics. |
| 🌐 **HNSW Index** | Multi-layer graph index (Malkov & Yashunin 2016) with soft-deletion tombstoning and graph preservation. |
| 🚀 **Parallel Flat Index** | OpenMP lock-free parallel brute-force vector search for 100% recall requirements. |
| 📑 **Paged Indexing** | Disk-backed `PagedIndex` with configurable LRU memory-mapped paging for datasets exceeding RAM. |
| 🛡️ **WAL & Recovery** | Write-Ahead Log (WAL) ensuring persistence and crash recovery. |
| 🔍 **Metadata Filtering** | Complex query-time JSON predicate evaluation (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`). |
| 🐍 **Python Interface** | Zero-copy NumPy integration powered by pybind11. |
| 🛠️ **CLI Suite** | Command-line benchmarking and search execution tool. |

---

## 🛠️ Build & Installation

### Prerequisites
- **Compiler**: MSVC (Visual Studio 2022+), GCC 10+, or Clang 12+
- **Build System**: CMake ≥ 3.20
- **Dependencies**: OpenMP, vcpkg (with `google-benchmark`, `gtest`, `pybind11`, `nlohmann-json`)

### Building from Source

```powershell
# 1. Configure CMake
cmake -B build -S . `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# 2. Build binaries in parallel
cmake --build build --config Release --parallel

# 3. Run unit tests
ctest --test-dir build -C Release --output-on-failure
```

---

## 💡 Quick Start & Usage

### C++ API Example

```cpp
#include "hnsw.hpp"
#include "vector_store.hpp"
#include <iostream>
#include <vector>

int main() {
    size_t dim = 128;
    neuraldb::HNSWConfig cfg;
    cfg.M = 16;
    cfg.ef_construction = 200;
    cfg.ef_search = 50;

    neuraldb::HNSWIndex index(dim, cfg);

    // Insert vector
    std::vector<float> vec(dim, 0.5f);
    index.insert(101, vec, R"({"category": "tech", "year": 2024})");

    // Search top-k
    auto results = index.search(vec.data(), 5);
    for (const auto& [score, id] : results) {
        std::cout << "ID: " << id << " | Similarity: " << score << "\n";
    }

    return 0;
}
```

### Python API Example

```python
import neuraldb
import numpy as np

# Initialize index
db = neuraldb.Index(dim=768, metric="cosine", M=16, ef_search=50)

# Zero-copy NumPy insertion
vector = np.random.randn(768).astype(np.float32)
db.insert(id=1, vector=vector, metadata={"category": "science", "year": 2023})

# Query with JSON metadata filter
results = db.search(
    query=vector,
    k=10,
    where={
        "year": {"$gte": 2020, "$lte": 2024},
        "category": {"$in": ["science", "tech"]}
    }
)
print("Results:", results)

# Soft delete vector
db.delete(id=1)
assert db.is_deleted(id=1)

# Compact tombstones & persist
db.compact("index.ndb", "wal.log")
```

### CLI Benchmarking

```powershell
# Run synthetic benchmark
.\build\Release\neuraldb_cli.exe bench --dim 768 --n 100000 --k 10

# Run benchmark on custom embeddings
.\build\Release\neuraldb_cli.exe bench --embeddings embeddings.npy --queries queries.npy --k 10
```

---

## 📊 Performance Benchmarks

| Index Type | Benchmark Configuration | Latency / Query | Recall@10 | Notes |
|---|---|---|---|---|
| **HNSW (AVX2)** | 100k vectors, dim=768, M=16 | ~1.0 ms | **0.96** | Lock-free candidate traversal |
| **Flat Index (AVX2)** | 100k vectors, dim=768 | ~12.5 ms | **1.00** | OpenMP parallelized brute-force |
| **Paged Index** | 100k vectors, dim=768 (LRU cache) | ~1.8 ms | **0.95** | Out-of-RAM disk paging overhead ~1.8x |

---

## 📁 Repository Architecture

```
NeuralDB/
├── include/              # Public C++ headers
│   ├── distance.hpp      # Distance kernel declarations & SIMD dispatch
│   ├── filter.hpp        # JSON predicate query parser & evaluator
│   ├── hnsw.hpp          # HNSW index implementation
│   ├── paged_index.hpp   # Disk-backed paged index & LRU cache
│   ├── storage.hpp       # Binary storage & MMap utils
│   ├── vector_store.hpp  # Flat index & SearchResult types
│   └── wal.hpp           # Write-Ahead Logging & crash recovery
├── src/                  # Core engine implementations
├── kernels/              # SIMD Distance Kernels (AVX2, NEON, Naive)
├── python/               # pybind11 C++ python module bindings
├── benchmarks/           # Google Benchmark suites
├── tests/                # Google Test suite
├── scripts/              # Helper utility scripts
└── CMakeLists.txt        # Master CMake build file
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
