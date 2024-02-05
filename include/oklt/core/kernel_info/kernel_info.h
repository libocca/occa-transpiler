#pragma once

#include <string>
#include <vector>
#include <optional>
#include <nlohmann/json.hpp>

namespace oklt {

enum struct DatatypeCategory {
  BUILTIN,
  CUSTOM,
};

//// map TaskState values to JSON as strings
//NLOHMANN_JSON_SERIALIZE_ENUM(
//  DatatypeCategory, {
//  {DatatypeCategory::BUILTIN, "builtin"},
//  {DatatypeCategory::CUSTOM, "custom"},
//});

struct DataType {
  std::string name;
  DatatypeCategory type;
  int bytes;
};

struct ArgumentInfo {
  bool is_const;
  DataType dtype;
  std::string name;
  bool is_ptr;
};

struct ParsedKernelInfo {
  std::vector<ArgumentInfo> arguments;
  std::string name;
  //INFO: for launcher template generation only
  int dimOuter = 0;
  int dimInner = 0;
  int tileSize = 0;
};

struct DependeciesInfo
{};

struct PropertyInfo {
  std::string compiler;
  std::string compiler_flags;
  std::string hash;
  std::string mode;
  bool verbose;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(PropertyInfo,
                                 compiler,
                                 compiler_flags,
                                 hash,
                                 mode,
                                 verbose);
};

struct KernelMetadata {
  std::optional<DependeciesInfo> dependencies = std::nullopt;
  std::string hash;
  std::vector<ParsedKernelInfo> metadata;
  std::optional<PropertyInfo> props = std::nullopt;
};

//INFO: because of different behaviour of presense field bytes when category is builtin
void to_json(nlohmann::json& json, const DataType& dt);
void from_json(const nlohmann::json& json, DataType& dt);

//INFO: needs custom function because of `const` field name
void to_json(nlohmann::json& json, const ArgumentInfo& argInfo);
void from_json(const nlohmann::json& json, ArgumentInfo& argInfo);

//INFO: skip some fields in serialization/deserialization process
void to_json(nlohmann::json& json, const ParsedKernelInfo& kernelMeta);
void from_json(const nlohmann::json& json, ParsedKernelInfo& kernelMeta);

//INFO: using optional to be able to have empty json object
void to_json(nlohmann::json& json, const KernelMetadata& kernelInfo);
void from_json(const nlohmann::json& json, KernelMetadata& kernelInfo);

}
