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

struct DatatypeInfo {
  std::string name;
  DatatypeCategory type;
  int bytes;
};

struct ArgumentInfo {
  bool is_const;
  DatatypeInfo dtype;
  std::string name;
  bool is_ptr;
};

struct KernelMetaInfo {
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

struct KernelInfo {
  std::optional<DependeciesInfo> dependencies = std::nullopt;
  std::string hash;
  std::vector<KernelMetaInfo> metadata;
  std::optional<PropertyInfo> props = std::nullopt;
};

//INFO: because of different behaviour of presense field bytes when category is builtin
void to_json(nlohmann::json& json, const DatatypeInfo& dt);
void from_json(const nlohmann::json& json, DatatypeInfo& dt);

//INFO: needs custom function because of `const` field name
void to_json(nlohmann::json& json, const ArgumentInfo& argInfo);
void from_json(const nlohmann::json& json, ArgumentInfo& argInfo);

//INFO: skip some fields in serialization/deserialization process
void to_json(nlohmann::json& json, const KernelMetaInfo& kernelMeta);
void from_json(const nlohmann::json& json, KernelMetaInfo& kernelMeta);

//INFO: using optional to be able to have empty json object
void to_json(nlohmann::json& json, const KernelInfo& kernelInfo);
void from_json(const nlohmann::json& json, KernelInfo& kernelInfo);

}
