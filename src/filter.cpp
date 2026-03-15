#include "filter.hpp"
#include <stdexcept>

namespace neuraldb {

namespace {

// Compare two JSON values (both numeric or both string).
// Returns -1/0/1 like strcmp.
int json_compare(const nlohmann::json& field_val, const nlohmann::json& filter_val) {
    if (field_val.is_number() && filter_val.is_number()) {
        double a = field_val.get<double>();
        double b = filter_val.get<double>();
        if (a < b) return -1;
        if (a > b) return  1;
        return 0;
    }
    if (field_val.is_string() && filter_val.is_string()) {
        auto a = field_val.get<std::string>();
        auto b = filter_val.get<std::string>();
        if (a < b) return -1;
        if (a > b) return  1;
        return 0;
    }
    // Mixed types: compare as strings
    return field_val.dump() == filter_val.dump() ? 0 : -1;
}

bool apply_op(const nlohmann::json& field_val, FilterOp op, const nlohmann::json& filter_val) {
    switch (op) {
    case FilterOp::EQ:  return json_compare(field_val, filter_val) == 0;
    case FilterOp::NE:  return json_compare(field_val, filter_val) != 0;
    case FilterOp::GT:  return json_compare(field_val, filter_val) >  0;
    case FilterOp::GTE: return json_compare(field_val, filter_val) >= 0;
    case FilterOp::LT:  return json_compare(field_val, filter_val) <  0;
    case FilterOp::LTE: return json_compare(field_val, filter_val) <= 0;
    case FilterOp::IN:
        if (!filter_val.is_array()) return false;
        for (auto& elem : filter_val)
            if (json_compare(field_val, elem) == 0) return true;
        return false;
    case FilterOp::NIN:
        if (!filter_val.is_array()) return true;
        for (auto& elem : filter_val)
            if (json_compare(field_val, elem) == 0) return false;
        return true;
    }
    return false;
}

} // anonymous namespace

WhereClause parse_where(const nlohmann::json& where_json) {
    WhereClause result;
    if (!where_json.is_object()) return result;

    static const std::unordered_map<std::string, FilterOp> op_map = {
        {"$eq",  FilterOp::EQ},
        {"$ne",  FilterOp::NE},
        {"$gt",  FilterOp::GT},
        {"$gte", FilterOp::GTE},
        {"$lt",  FilterOp::LT},
        {"$lte", FilterOp::LTE},
        {"$in",  FilterOp::IN},
        {"$nin", FilterOp::NIN},
    };

    for (auto& [field, spec] : where_json.items()) {
        std::vector<FieldFilter> filters;
        if (spec.is_object()) {
            // e.g. {"$gte": 2020, "$lte": 2024}
            for (auto& [op_key, val] : spec.items()) {
                auto it = op_map.find(op_key);
                if (it == op_map.end())
                    throw std::invalid_argument("Unknown filter operator: " + op_key);
                filters.push_back({it->second, val});
            }
        } else {
            // Bare value → treat as $eq
            filters.push_back({FilterOp::EQ, spec});
        }
        result[field] = std::move(filters);
    }
    return result;
}

bool evaluate_filter(const std::string& metadata_json, const WhereClause& where) {
    if (where.empty()) return true;

    nlohmann::json meta;
    try {
        meta = nlohmann::json::parse(metadata_json);
    } catch (...) {
        return false; // unparseable metadata fails all filters
    }
    if (!meta.is_object()) return false;

    for (auto& [field, filters] : where) {
        if (!meta.contains(field)) return false; // field absent → no match
        const auto& field_val = meta[field];
        for (auto& ff : filters) {
            if (!apply_op(field_val, ff.op, ff.value)) return false;
        }
    }
    return true;
}

} // namespace neuraldb
