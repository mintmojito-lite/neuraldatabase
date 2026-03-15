cmake_minimum_required(VERSION 3.20)
project(NeuralDB VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Architecture detection & compiler flags ────────────────────────────────────
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /openmp /W3 /EHsc /MP)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
else()
    add_compile_options(-O3 -fopenmp -Wall -Wextra)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm|ARM)")
        # ARM: enable NEON
        add_compile_options(-mfpu=neon -mfloat-abi=hard)
        add_compile_definitions(NEURALDB_NEON)
        message(STATUS "NeuralDB: ARM NEON enabled")
    else()
        # x86: enable AVX2 + FMA
        add_compile_options(-mavx2 -mfma)
        message(STATUS "NeuralDB: AVX2+FMA enabled")
    endif()
endif()

# ── vcpkg packages ─────────────────────────────────────────────────────────────
find_package(benchmark CONFIG REQUIRED)   # vcpkg package: benchmark
find_package(GTest CONFIG REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenMP REQUIRED)

# ── neuraldb_core (static library) ─────────────────────────────────────────────
add_library(neuraldb_core STATIC
    src/distance.cpp
    src/vector_store.cpp
    src/hnsw.cpp
    src/storage.cpp
    src/wal.cpp
    src/filter.cpp
    src/paged_index.cpp
    kernels/cosine_naive.cpp
    kernels/cosine_avx2.cpp
    kernels/l2_avx2.cpp
    kernels/cosine_neon.cpp
    kernels/l2_neon.cpp
)

target_include_directories(neuraldb_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(neuraldb_core PUBLIC
    OpenMP::OpenMP_CXX
    nlohmann_json::nlohmann_json
)

# ── benchmarks ─────────────────────────────────────────────────────────────────
add_executable(bench_search benchmarks/bench_search.cpp)
target_link_libraries(bench_search PRIVATE neuraldb_core benchmark::benchmark benchmark::benchmark_main)
target_include_directories(bench_search PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── unit tests ─────────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_distance tests/test_distance.cpp)
target_link_libraries(test_distance PRIVATE neuraldb_core GTest::gtest GTest::gtest_main)
target_include_directories(test_distance PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_hnsw tests/test_hnsw.cpp)
target_link_libraries(test_hnsw PRIVATE neuraldb_core GTest::gtest GTest::gtest_main)
target_include_directories(test_hnsw PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_deletes tests/test_deletes.cpp)
target_link_libraries(test_deletes PRIVATE neuraldb_core GTest::gtest GTest::gtest_main)
target_include_directories(test_deletes PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_neon tests/test_neon.cpp)
target_link_libraries(test_neon PRIVATE neuraldb_core GTest::gtest GTest::gtest_main)
target_include_directories(test_neon PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_paged tests/test_paged.cpp)
target_link_libraries(test_paged PRIVATE neuraldb_core GTest::gtest GTest::gtest_main)
target_include_directories(test_paged PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_executable(test_filter tests/test_filter.cpp)
target_link_libraries(test_filter PRIVATE neuraldb_core GTest::gtest GTest::gtest_main)
target_include_directories(test_filter PRIVATE ${CMAKE_SOURCE_DIR}/include)

add_test(NAME DistanceTests COMMAND test_distance)
add_test(NAME HNSWTests     COMMAND test_hnsw)
add_test(NAME DeleteTests   COMMAND test_deletes)
add_test(NAME NEONTests     COMMAND test_neon)
add_test(NAME PagedTests    COMMAND test_paged)
add_test(NAME FilterTests   COMMAND test_filter)

# ── CLI tool ───────────────────────────────────────────────────────────────────
add_executable(neuraldb_cli src/cli.cpp)
target_link_libraries(neuraldb_cli PRIVATE neuraldb_core OpenMP::OpenMP_CXX)
target_include_directories(neuraldb_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)

# ── Python bindings ────────────────────────────────────────────────────────────
pybind11_add_module(neuraldb python/bindings.cpp)
target_link_libraries(neuraldb PRIVATE neuraldb_core)
target_include_directories(neuraldb PRIVATE ${CMAKE_SOURCE_DIR}/include)
