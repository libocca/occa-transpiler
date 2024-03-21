#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>
#include <list>

namespace oklt {

enum struct DatatypeCategory {
    BUILTIN,
    CUSTOM,
    STRUCT,
    TUPLE,
};

struct StructFieldInfo;
struct TupleElementDataType;

struct TupleElementDataType {
    int64_t tupleSize = -1;
    DatatypeCategory typeCategory = DatatypeCategory::CUSTOM;
    std::string name;                                           // Used only for builtin and custom
    std::list<StructFieldInfo> fields;                          // Used only for struct element type
    std::shared_ptr<TupleElementDataType> tupleElementDType = nullptr;    // Used only for tuple element type
};

struct DataType {
    std::string name;
    DatatypeCategory typeCategory;
    int bytes = 0;                              // used only for custom
    std::list<StructFieldInfo> fields;          // used only for structs
    std::shared_ptr<TupleElementDataType> tupleElementDType;     // used only fot tuples
};

struct StructFieldInfo {
    DataType dtype;
    std::string name;
};

struct ArgumentInfo {
    bool is_const;
    DataType dtype;
    std::string name;
    bool is_ptr;
};

enum class LoopType {
    Regular,
    Inner,
    Outer,
};

inline std::string toString(LoopType lmd) {
    switch (lmd) {
        case LoopType::Regular:
            return "regular";
        case LoopType::Outer:
            return "outer";
        case LoopType::Inner:
            return "inner";
    }
    return "<uknown>";
}

// TODO replace clang types by own
enum class BinOp { Eq, Le, Lt, Gt, Ge, AddAssign, RemoveAssign, Other };

enum class UnOp { PreInc, PostInc, PreDec, PostDec, Other };

struct KernelInfo {
    std::string name;
    std::vector<ArgumentInfo> args;
};

struct DependeciesInfo {};

struct PropertyInfo {
    std::string compiler;
    std::string compiler_flags;
    std::string hash;
    std::string mode;
    bool verbose;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(PropertyInfo, compiler, compiler_flags, hash, mode, verbose);
};

struct ProgramMetaData {
    std::optional<DependeciesInfo> dependencies = std::nullopt;
    std::string hash;
    std::list<KernelInfo> kernels;
    std::optional<PropertyInfo> props = std::nullopt;
};

// INFO: because of different behaviour of presence field bytes when category is builtin
void to_json(nlohmann::json& json, const DataType& dt);
void from_json(const nlohmann::json& json, DataType& dt);

// INFO: needs custom function because of `const` field name
void to_json(nlohmann::json& json, const ArgumentInfo& argInfo);
void from_json(const nlohmann::json& json, ArgumentInfo& argInfo);

// INFO: skip some fields in serialization/deserialization process
void to_json(nlohmann::json& json, const KernelInfo& kernelMeta);
void from_json(const nlohmann::json& json, KernelInfo& kernelMeta);

// INFO: using optional to be able to have empty json object
void to_json(nlohmann::json& json, const ProgramMetaData& programMeta);
void from_json(const nlohmann::json& json, ProgramMetaData& programMeta);

}  // namespace oklt
