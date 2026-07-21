# NeuralDB

NeuralDB is a lightweight, high-performance vector database engine written in C++17 with Python bindings.

Designed strictly from first principles without external ANN libraries (such as FAISS or Annoy), NeuralDB demonstrates systems-level optimizations including SIMD vector dispatch (AVX2 / ARM NEON), lock-free parallel graph traversals, mmap-backed disk paging, and Write-Ahead Logging (WAL) for durability.

---

## Architectural Overview

### Features

| Component | Engineering Details |
|---|---|
| **SIMD Distance Kernels** | Runtime-dispatched AVX2 (x86_64) and ARM NEON kernels evaluating Cosine similarity and L2 distance. |
| **HNSW Graph Index** | Multi-layer graph implementation based on Malkov & Yashunin (2016). Supports soft deletion via tombstoning without compromising graph connectivity. |
| **Parallel Flat Index** | OpenMP-accelerated brute-force vector search offering zero recall degradation ($100\%$ accuracy). |
| **Paged Indexing** | Memory-mapped out-of-RAM storage via `PagedIndex` with configurable LRU cache management. |
| **WAL & Crash Recovery** | Append-only Write-Ahead Log (`WAL`) ensuring persistence and atomic compaction checkpoints. |
| **Metadata Filtering** | In-memory evaluation of arbitrary JSON predicate expressions (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`) during graph traversal and brute-force scans. |
| **Python Bindings** | C++ bindings via pybind11 supporting zero-copy NumPy array ingestion and query filtering. |
| **CLI Tooling** | `neuraldb_cli` utility for generating synthetic embeddings, running throughput benchmarks, and performing index evaluation. |

---

## Building and Installation

### Prerequisites

- **Compiler**: MSVC (Visual Studio 2022+), GCC 10+, or Clang 12+ supporting C++17.
- **Build System**: CMake $\ge 3.20$.
- **Dependencies**: OpenMP runtime and vcpkg packages (`google-benchmark`, `gtest`, `pybind11`, `nlohmann-json`).

### Build Instructions

```powershell
# Configure project
cmake -B build -S . `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Compile binaries
cmake --build build --config Release --parallel

# Execute test suite
ctest --test-dir build -C Release --output-on-failure
```

---

## Usage Examples

### C++ Interface

```cpp
#include "hnsw.hpp"
#include "vector_store.hpp"
#include <iostream>
#include <vector>

int main() {
    constexpr size_t dim = 128;
    neuraldb::HNSWConfig config;
    config.M = 16;
    config.ef_construction = 200;
    config.ef_search = 50;

    neuraldb::HNSWIndex index(dim, config);

    // Insert vector with JSON metadata
    std::vector<float> vec(dim, 0.5f);
    index.insert(101, vec, R"({"category": "tech", "year": 2024})");

    // Perform approximate nearest neighbor search
    auto results = index.search(vec.data(), 5);
    for (const auto& [score, id] : results) {
        std::cout << "Vector ID: " << id << " | Similarity: " << score << "\n";
    }

    return 0;
}
```

### Python Interface

```python
import neuraldb
import numpy as np

# Initialize HNSW index
db = neuraldb.Index(dim=768, metric="cosine", M=16, ef_search=50)

# Insert vector with zero-copy NumPy array
vector = np.random.randn(768).astype(np.float32)
db.insert(id=1, vector=vector, metadata={"category": "science", "year": 2023})

# Query index with JSON metadata predicate filtering
results = db.search(
    query=vector,
    k=10,
    where={
        "year": {"$gte": 2020, "$lte": 2024},
        "category": {"$in": ["science", "tech"]}
    }
)
print("Query Results:", results)

# Perform soft delete and index compaction
db.delete(id=1)
db.compact("index.ndb", "wal.log")
```

### Command Line Interface

```powershell
# Run benchmark with synthetic vectors (N=100000, dim=768, k=10)
.\build\Release\neuraldb_cli.exe bench --dim 768 --n 100000 --k 10

# Run evaluation with custom NumPy dataset files
.\build\Release\neuraldb_cli.exe bench --embeddings embeddings.npy --queries queries.npy --k 10
```

---

## Performance Metrics

| Index Configuration | Dataset Parameters | Query Latency | Recall@10 | Operational Characteristics |
|---|---|---|---|---|
| **HNSW (AVX2)** | 100k vectors, dim=768, M=16 | ~1.0 ms | **0.96** | Lock-free beam search traversal |
| **Flat Index (AVX2)** | 100k vectors, dim=768 | ~12.5 ms | **1.00** | Lock-free OpenMP parallel scan |
| **Paged Index** | 100k vectors, dim=768 (LRU cache) | ~1.8 ms | **0.95** | Out-of-RAM disk paging (~1.8x overhead) |

---

## Repository Layout

```
NeuralDB/
├── include/              # Public C++ interface headers
│   ├── distance.hpp      # SIMD distance kernel declarations
│   ├── filter.hpp        # JSON predicate filter parser and evaluator
│   ├── hnsw.hpp          # HNSW graph index implementation
│   ├── paged_index.hpp   # Memory-mapped disk index & LRU page cache
│   ├── storage.hpp       # Binary serialization & mmap utilities
│   ├── vector_store.hpp  # FlatIndex vector store definitions
│   └── wal.hpp           # Write-Ahead Log implementation
├── src/                  # Core engine source files
├── kernels/              # SIMD Distance Kernels (AVX2, NEON, Naive)
├── python/               # pybind11 C++ module bindings
├── benchmarks/           # Google Benchmark implementation
├── tests/                # Google Test suites
├── scripts/              # Dataset generation helper scripts
└── CMakeLists.txt        # Master CMake build configuration
```

---

## License

Distributed under the MIT License. See `LICENSE` for details.
