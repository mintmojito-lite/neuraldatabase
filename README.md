# NeuralDB

A from-scratch, high-performance vector database engine in C++17.

> Pure systems engineering — no FAISS, no Annoy, no ANN libraries.

## Features

| Feature | Detail |
|---|---|
| **Distance kernels** | AVX2 & ARM NEON runtime-dispatched cosine & L2 |
| **Flat index** | OpenMP-parallelised brute-force, O(N) |
| **HNSW index** | Malkov & Yashunin 2016, multi-layer graph, soft deletion via tombstones |
| **Persistence & Paging** | `PagedIndex` for out-of-RAM support via mmap, WAL crash recovery |
| **Metadata filtering** | Complex JSON predicates (`$gte`, `$in`, `$ne`, etc.) evaluated at query-time |
| **Python bindings** | pybind11, zero-copy NumPy arrays, `where={}` filter support |
| **CLI** | `neuraldb_cli bench` prints full benchmark table |

## Build

### Prerequisites
- Windows, MSVC 2022+, CMake ≥ 3.20
- vcpkg with: `google-benchmark gtest pybind11 nlohmann-json`

```powershell
# Configure
cmake -B build -S . `
  -DCMAKE_BUILD_TYPE=Release `
  -DCMAKE_TOOLCHAIN_FILE="$env:VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"

# Build
cmake --build build --config Release --parallel

# Test
ctest --test-dir build -C Release --output-on-failure

# Benchmark
.\build\Release\bench_search.exe

# CLI
.\build\Release\neuraldb_cli.exe bench --dim 768 --n 100000 --k 10
```

## Python

```python
import neuraldb
import numpy as np

db = neuraldb.Index(dim=768, metric="cosine", M=16, ef_search=50)

# Zero-copy NumPy insert
vec = np.random.randn(768).astype(np.float32)
db.insert(id=1, vector=vec, metadata={"text": "hello world"})

# Search with complex JSON filtering
results = db.search(
    query=vec,
    k=10,
    where={
        "year": {"$gte": 2020, "$lte": 2024},
        "category": {"$in": ["science", "tech"]}
    }
)
# [{"id": 1, "score": 1.0, "metadata": {"year": 2023, "category": "science"}}, ...]

# Soft deletion
db.delete(id=1)
assert db.is_deleted(id=1)

# Compact the index (remove tombstones) & save
db.compact("my_index.ndb", "wal_checkpoint.log")
```

## Benchmark Results

| Configuration | Metric | Detail |
|---|---|---|
| **HNSW Search (x86 AVX2)** | ~1.0 ms / query | Recall@10: 0.96 (100k vectors, dim=768) |
| **HNSW Search (ARM NEON)** | ~1.5 ms / query | *Estimated runtime dispatched on ARM* |
| **Dynamic Deletes** | O(1) latency | Constant time marking of tombstone bits |
| **PagedIndex Overhead** | ~1.3x - 2.5x | Disk reads vs pure in-memory depends heavily on LRU hit rate |

*Run `neuraldb_cli bench --dim 768 --n 100000 --k 10` to regenerate full throughput tables.*

## Architecture

```
neuraldb/
├── include/          # Public headers
│   ├── distance.hpp  # Kernel declarations
│   ├── vector_store.hpp  # FlatIndex + SearchResult
│   ├── hnsw.hpp      # HNSWIndex
│   ├── storage.hpp   # MMapReader, FileWriter
│   └── wal.hpp       # Write-ahead log
├── src/              # Core implementations
├── kernels/          # cosine_naive, cosine_avx2, l2_avx2
├── benchmarks/       # Google Benchmark
├── tests/            # Google Test
└── python/           # pybind11 bindings
```

## Algorithm (HNSW)

Based on [Malkov & Yashunin 2016](https://arxiv.org/abs/1603.09320):
- **Layer assignment**: `floor(-ln(U(0,1)) * mL)`, `mL = 1/ln(M)`
- **Insert**: greedy descent to assigned layer, beam search with `ef_construction`, M bidirectional links per layer
- **Search**: greedy descent to layer 0, beam search with `ef_search`, return top-k
- **Default params**: M=16, ef_construction=200, ef_search=50

## License

MIT
