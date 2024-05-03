#pragma once

#include <nlohmann/json.hpp>

#include <list>
#include <optional>
#include <string>
#include <vector>

namespace oklt {

/**
 * @brief Enum for the categories of data types in the metadata.
 */
enum struct DatatypeCategory {
    BUILTIN,  ///< Built-in data type.
    CUSTOM,   ///< Custom data type.
    STRUCT,   ///< Struct data type.
    TUPLE,    ///< Tuple data type.
    ENUM,     ///< Enum data type.
};

struct StructFieldInfo;
struct TupleElementDataType;

/**
 * @brief Represents a data type in the metadata.
 */
struct DataType {
    std::string name;               ///< The name of the data type.
    DatatypeCategory typeCategory;  ///< The category of the data type.
    int bytes = 0;  ///< The size of the data type in bytes. Used only for custom data types.
    std::list<StructFieldInfo>
        fields;  ///< The fields of the struct. Used only for struct data types.
    std::shared_ptr<TupleElementDataType>
        tupleElementDType;  ///< The data type of the tuple element. Used only for tuple data types.
    std::vector<std::string>
        enumNames;  ///< The names of the enum values. Used only for enum data types.
};

/**
 * @brief Represents the data type of an element in a tuple.
 */
struct TupleElementDataType {
    int64_t tupleSize = -1;  ///< The size of the tuple.
    DataType elementDType;   ///< The data type of the element.
};

/**
 * @brief Represents a field in a struct or class
 */
struct StructFieldInfo {
    DataType dtype;    ///< The data type of the field.
    std::string name;  ///< The name of the field.
};

/**
 * @brief Represents an argument in a kernel function.
 */
struct ArgumentInfo {
    bool is_const;     ///< Whether the argument is constant.
    DataType dtype;    ///< The data type of the argument.
    std::string name;  ///< The name of the argument.
    bool is_ptr;       ///< Whether the argument is a pointer.
};

/**
 * @brief Enum for the types of loops.
 */
enum class LoopType {
    Regular,  ///< Regular loop.
    Inner,    ///< Inner loop (@inner).
    Outer,    ///< Outer loop (@outer).
};

/**
 * @brief Converts a LoopType to a string.
 *
 * @param lmd The LoopType to convert.
 * @return std::string The string representation of the LoopType.
 */
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

/**
 * @brief Enum for binary operations.
 */
enum class BinOp {
    Eq,            ///< Equal to.
    Le,            ///< Less than or equal to.
    Lt,            ///< Less than.
    Gt,            ///< Greater than.
    Ge,            ///< Greater than or equal to.
    AddAssign,     ///< Addition assignment (+=).
    RemoveAssign,  ///< Removal assignment.
    Other          ///< Other binary operation.
};

/**
 * @brief Enum for unary operations.
 */
enum class UnOp {
    PreInc,   ///< Pre-increment.
    PostInc,  ///< Post-increment.
    PreDec,   ///< Pre-decrement.
    PostDec,  ///< Post-decrement.
    Other     ///< Other unary operation.
};

/**
 * @brief Represents a kernel function.
 */
struct KernelInfo {
    std::string name;                ///< The name of the kernel function.
    std::vector<ArgumentInfo> args;  ///< The arguments of the kernel function.
};

/**
 * @brief Represents the dependencies of a program.
 */
struct DependeciesInfo {};

/**
 * @brief Represents some of the properties of a metadata.
 */
struct PropertyInfo {
    std::string compiler;
    std::string compiler_flags;
    std::string hash;
    std::string mode;
    bool verbose;
    NLOHMANN_DEFINE_TYPE_INTRUSIVE(PropertyInfo, compiler, compiler_flags, hash, mode, verbose);
};

/**
 * @brief Represents the metadata of a program.
 */
struct ProgramMetaData {
    std::optional<DependeciesInfo> dependencies = std::nullopt;
    std::string hash;
    std::list<KernelInfo> kernels;
    std::optional<PropertyInfo> props = std::nullopt;
};

// INFO: because of different behaviour of presence field bytes when category is builtin
/**
 * @brief Converts a DataType to a JSON object.
 *
 * @param json The JSON object to convert to.
 * @param dt The DataType to convert.
 */
void to_json(nlohmann::json& json, const DataType& dt);
/**
 * @brief Converts a JSON object to a DataType.
 *
 * @param json The JSON object to convert from.
 * @param dt The DataType to convert to.
 */
void from_json(const nlohmann::json& json, DataType& dt);

// INFO: needs custom function because of `const` field name
/**
 * @brief Converts an ArgumentInfo to a JSON object.
 *
 * @param json The JSON object to convert to.
 * @param argInfo The ArgumentInfo to convert.
 */
void to_json(nlohmann::json& json, const ArgumentInfo& argInfo);

/**
 * @brief Converts a JSON object to an ArgumentInfo.
 *
 * @param json The JSON object to convert from.
 * @param argInfo The ArgumentInfo to convert to.
 */
void from_json(const nlohmann::json& json, ArgumentInfo& argInfo);

// INFO: skip some fields in serialization/deserialization process
/**
 * @brief Converts a KernelInfo to a JSON object.
 *
 * @param json The JSON object to convert to.
 * @param kernelMeta The KernelInfo to convert.
 */
void to_json(nlohmann::json& json, const KernelInfo& kernelMeta);

/**
 * @brief Converts a JSON object to a KernelInfo.
 *
 * @param json The JSON object to convert from.
 * @param kernelMeta The KernelInfo to convert to.
 */
void from_json(const nlohmann::json& json, KernelInfo& kernelMeta);

// INFO: using optional to be able to have empty json object
/**
 * @brief Converts a ProgramMetaData to a JSON object.
 *
 * @param json The JSON object to convert to.
 * @param programMeta The ProgramMetaData to convert.
 */
void to_json(nlohmann::json& json, const ProgramMetaData& programMeta);

/**
 * @brief Converts a JSON object to a ProgramMetaData.
 *
 * @param json The JSON object to convert from.
 * @param programMeta The ProgramMetaData to convert to.
 */
void from_json(const nlohmann::json& json, ProgramMetaData& programMeta);

}  // namespace oklt
