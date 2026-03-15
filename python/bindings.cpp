#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "hnsw.hpp"
#include "vector_store.hpp"
#include "filter.hpp"
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace neuraldb;
using json = nlohmann::json;

// ── Python object → JSON ──────────────────────────────────────────────────────
/// Deep-converts a Python dict/list to nlohmann::json.
/// Handles nested dicts (including $-operator sub-dicts).
static json py_to_json(const py::object& obj) {
    if (obj.is_none()) return json(nullptr);
    if (py::isinstance<py::bool_>(obj))  return obj.cast<bool>();
    if (py::isinstance<py::int_>(obj))   return obj.cast<int64_t>();
    if (py::isinstance<py::float_>(obj)) return obj.cast<double>();
    if (py::isinstance<py::str>(obj))    return obj.cast<std::string>();
    if (py::isinstance<py::list>(obj)) {
        json arr = json::array();
        for (auto& item : obj.cast<py::list>()) arr.push_back(py_to_json(item.cast<py::object>()));
        return arr;
    }
    if (py::isinstance<py::dict>(obj)) {
        json j = json::object();
        for (auto& kv : obj.cast<py::dict>()) {
            std::string key = py::str(kv.first);
            j[key] = py_to_json(kv.second.cast<py::object>());
        }
        return j;
    }
    return obj.cast<std::string>();
}

// Shallow dict → JSON string (for metadata)
static std::string dict_to_json(const py::object& obj) {
    if (obj.is_none()) return "{}";
    return py_to_json(obj).dump();
}

// Get float* from numpy array (zero-copy)
static const float* get_ptr(const py::array_t<float>& arr, size_t expected_dim) {
    auto buf = arr.request();
    if (buf.ndim != 1 || static_cast<size_t>(buf.shape[0]) != expected_dim)
        throw std::invalid_argument("Expected 1D float32 array of size " + std::to_string(expected_dim));
    return static_cast<const float*>(buf.ptr);
}

// Build a python dict from a JSON metadata string
static py::dict json_meta_to_dict(const std::string& meta) {
    py::dict md;
    try {
        auto j = json::parse(meta);
        for (auto& [k2, v] : j.items()) {
            if (v.is_string())           md[py::str(k2)] = v.get<std::string>();
            else if (v.is_number_integer()) md[py::str(k2)] = v.get<int64_t>();
            else if (v.is_number_float())   md[py::str(k2)] = v.get<double>();
            else if (v.is_boolean())        md[py::str(k2)] = v.get<bool>();
            else                            md[py::str(k2)] = v.dump();
        }
    } catch (...) {
        md[py::str("_raw")] = meta;
    }
    return md;
}

PYBIND11_MODULE(neuraldb, m) {
    m.doc() = "NeuralDB — high-performance vector database with HNSW";

    // ── HNSWConfig ──────────────────────────────────────────────────────────
    py::class_<HNSWConfig>(m, "HNSWConfig")
        .def(py::init<>())
        .def_readwrite("M", &HNSWConfig::M)
        .def_readwrite("ef_construction", &HNSWConfig::ef_construction)
        .def_readwrite("ef_search", &HNSWConfig::ef_search);

    // ── Index (wraps HNSWIndex) ─────────────────────────────────────────────
    py::class_<HNSWIndex>(m, "Index")
        .def(py::init([](int dim, const std::string& metric, int M, int ef_construction, int ef_search){
            HNSWConfig cfg;
            cfg.M = M;
            cfg.ef_construction = ef_construction;
            cfg.ef_search = ef_search;
            if (metric != "cosine" && metric != "l2")
                throw std::invalid_argument("metric must be 'cosine' or 'l2'");
            return new HNSWIndex(static_cast<size_t>(dim), cfg);
        }),
        py::arg("dim"),
        py::arg("metric") = "cosine",
        py::arg("M") = 16,
        py::arg("ef_construction") = 200,
        py::arg("ef_search") = 50)

        .def("insert", [](HNSWIndex& self,
                          uint64_t id,
                          const py::array_t<float>& vec,
                          const py::object& metadata) {
            const float* ptr = get_ptr(vec, self.dim());
            self.insert(id, ptr, dict_to_json(metadata));
        },
        py::arg("id"),
        py::arg("vector"),
        py::arg("metadata") = py::none())

        .def("delete", [](HNSWIndex& self, uint64_t id) {
            self.delete_vector(id);
        }, py::arg("id"))

        .def("is_deleted", [](HNSWIndex& self, uint64_t id) {
            return self.is_deleted(id);
        }, py::arg("id"))

        .def("compact", [](HNSWIndex& self,
                           const std::string& index_path,
                           const std::string& wal_path) {
            self.compact(index_path, wal_path);
        }, py::arg("index_path"), py::arg("wal_path"))

        .def("search", [](HNSWIndex& self,
                          const py::array_t<float>& query,
                          int k,
                          const py::object& where) -> py::list {
            const float* ptr = get_ptr(query, self.dim());

            // Build WhereClause if provided
            WhereClause wc;
            bool has_where = !where.is_none();
            if (has_where) {
                json where_json = py_to_json(where);
                wc = parse_where(where_json);
            }

            auto raw_results = self.search(ptr, has_where ? k * 4 : k);

            py::list out;
            int returned = 0;
            for (auto& [score, id] : raw_results) {
                if (returned >= k) break;
                // Find internal id
                uint32_t iid = 0;
                for (uint32_t i = 0; i < static_cast<uint32_t>(self.size()); ++i) {
                    if (self.external_id(i) == id) { iid = i; break; }
                }
                const std::string& meta = self.metadata(iid);
                if (has_where && !evaluate_filter(meta, wc)) continue;

                py::dict d;
                d["id"]       = id;
                d["score"]    = score;
                d["metadata"] = json_meta_to_dict(meta);
                out.append(d);
                ++returned;
            }
            return out;
        },
        py::arg("query"),
        py::arg("k") = 10,
        py::arg("where") = py::none())

        .def("save", &HNSWIndex::save, py::arg("path"))
        .def("load", &HNSWIndex::load, py::arg("path"))
        .def("__len__",         &HNSWIndex::size)
        .def("live_size",       &HNSWIndex::live_size)
        .def("set_ef_search",   &HNSWIndex::set_ef_search, py::arg("ef"));
}
