#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>
#include <clang/AST/ParentMapContext.h>

#include <tl/expected.hpp>

namespace oklt {
using json = nlohmann::json;
void to_json(json& j, const DataType& dt) {
    if (dt.type == DatatypeCategory::BUILTIN) {
        j = json{{"name", dt.name}, {"type", "builtin"}};
    } else {
        j = json{{"name", "none"}, {"type", "custom"}, {"bytes", dt.bytes}};
    }
}

void from_json(const json& j, DataType& dt) {
    auto dtCategory = j.at("type").get<std::string>();
    if (dtCategory == "builtin") {
        dt.type = DatatypeCategory::BUILTIN;
        j.at("name").get_to(dt.name);
    } else {
        dt.type = DatatypeCategory::CUSTOM;
        dt.name = "none";
        j.at("bytes").get_to(dt.bytes);
    }
}

void to_json(json& j, const ArgumentInfo& argInfo) {
    j = json{{"const", argInfo.is_const},
             {"dtype", argInfo.dtype},
             {"name", argInfo.name},
             {"ptr", argInfo.is_ptr}};
}

void from_json(const json& j, ArgumentInfo& argInfo) {
    j.at("const").get_to(argInfo.is_const);
    j.at("dtype").get_to(argInfo.dtype);
    j.at("name").get_to(argInfo.name);
    j.at("ptr").get_to(argInfo.is_ptr);
}

void to_json(json& j, const KernelInfo& kernelMeta) {
    j = json{{"arguments", kernelMeta.args}, {"name", kernelMeta.name}};
}

void from_json(const json& j, KernelInfo& kernelMeta) {
    j.at("arguments").get_to(kernelMeta.args);
    j.at("name").get_to(kernelMeta.name);
}

void to_json(json& j, const ProgramMetaData& programMeta) {
    if (programMeta.props.has_value()) {
        j = json{
            {"dependencies", json::object()},  // INFO: always empty object, can't define the type
            {"hash", programMeta.hash},
            {"metadata", programMeta.kernels},
            {"props", programMeta.props.value()}};
    } else {
        j = json{
            {"dependencies", json::object()},  // INFO: always empty object, can't define the type
            {"hash", programMeta.hash},
            {"metadata", programMeta.kernels},
            {"props", json::object()}};
    }
}

void from_json(const json& j, ProgramMetaData& programMeta) {
    programMeta.dependencies = std::nullopt;
    const auto& value = j.at("props");
    if (value.is_object() && !value.empty()) {
        j.at("hash").get_to(programMeta.hash);
        j.at("metadata").get_to(programMeta.kernels);
        PropertyInfo prop;
        value.get_to(prop);
        programMeta.props = prop;
    } else {
        j.at("hash").get_to(programMeta.hash);
        j.at("metadata").get_to(programMeta.kernels);
        programMeta.props = std::nullopt;
    }
}
}  // namespace oklt
