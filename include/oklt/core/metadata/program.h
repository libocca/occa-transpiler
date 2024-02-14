#pragma once

#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <vector>

namespace clang {
struct FunctionDecl;
}
namespace oklt {

enum struct DatatypeCategory {
  BUILTIN,
  CUSTOM,
};

//// map TaskState values to JSON as strings
// NLOHMANN_JSON_SERIALIZE_ENUM(
//   DatatypeCategory, {
//   {DatatypeCategory::BUILTIN, "builtin"},
//   {DatatypeCategory::CUSTOM, "custom"},
// });

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

struct KernelInstance {
  // INFO: for launcher template generation only
  int dimOuter = 0;
  int dimInner = 0;
  int tileSize = 0;
};

struct KernelInfo {
  std::string name;
  std::vector<ArgumentInfo> args;
  std::vector<KernelInstance> instances;
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
  std::vector<KernelInfo> kernels;
  std::optional<PropertyInfo> props = std::nullopt;
  KernelInfo& addKernelInfo(std::string name, size_t numArg, size_t numKern) {
    // create new slot
    // there is no way to pass args nicely for POD
    kernels.emplace_back();

    // get just created slot and fill it
    auto& kiPtr = kernels.back();
    kiPtr.name = std::move(name);
    kiPtr.args.resize(numArg);
    kiPtr.instances.resize(numKern);

    return kiPtr;
  }
};

// INFO: because of different behaviour of presense field bytes when category is builtin
void to_json(nlohmann::json& json, const DataType& dt);
void from_json(const nlohmann::json& json, DataType& dt);

// INFO: needs custom function because of `const` field name
void to_json(nlohmann::json& json, const ArgumentInfo& argInfo);
void from_json(const nlohmann::json& json, ArgumentInfo& argInfo);

// INFO: skip some fields in serialization/deserialization process
void to_json(nlohmann::json& json, const KernelInfo& kernelMeta);
void from_json(const nlohmann::json& json, KernelInfo& kernelMeta);

// INFO: using optional to be able to have empty json object
void to_json(nlohmann::json& json, const ProgramMetaData& kernelInfo);
void from_json(const nlohmann::json& json, ProgramMetaData& kernelInfo);

}  // namespace oklt
