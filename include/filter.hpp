#pragma once
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

namespace neuraldb {

/// Supported filter operators.
enum class FilterOp { EQ, NE, GT, GTE, LT, LTE, IN, NIN };

/// A single (op, value) constraint on a field.
struct FieldFilter {
    FilterOp         op;
    nlohmann::json   value; // scalar or array (for $in/$nin)
};

/// Map of field_name → list of constraints (all must match — AND semantics).
using WhereClause = std::unordered_map<std::string, std::vector<FieldFilter>>;

/// Parse a where JSON object like:
///   { "year": {"$gte": 2020, "$lte": 2024},
///     "category": {"$in": ["science", "tech"]},
///     "score": {"$gt": 0.8} }
/// Returns a WhereClause that evaluate_filter() can use.
WhereClause parse_where(const nlohmann::json& where_json);

/// Evaluate whether a metadata JSON string satisfies all constraints.
/// Returns true iff every field/op in 'where' is satisfied.
bool evaluate_filter(const std::string& metadata_json, const WhereClause& where);

} // namespace neuraldb
