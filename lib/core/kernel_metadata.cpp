#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>
#include <clang/AST/ParentMapContext.h>

#include <tl/expected.hpp>

namespace oklt {
using json = nlohmann::json;

void to_json(json& j, const DatatypeCategory& cat) {
    switch (cat) {
        case DatatypeCategory::BUILTIN:
            j = "builtin";
            break;
        case DatatypeCategory::CUSTOM:
            j = "custom";
            break;
        case DatatypeCategory::STRUCT:
            j = "struct";
            break;
        case DatatypeCategory::TUPLE:
            j = "tuple";
            break;
        default:
            j = "";
    }
}

void from_json(const json& j, DatatypeCategory& cat) {
    if (j == "builtin") {
        cat = DatatypeCategory::BUILTIN;
    }
    if (j == "custom") {
        cat = DatatypeCategory::CUSTOM;
    }
    if (j == "struct") {
        cat = DatatypeCategory::STRUCT;
    }
    if (j == "tuple") {
        cat = DatatypeCategory::TUPLE;
    }
}

void to_json(json& j, const StructFieldInfo& dt) {
    j = json{{"name", dt.name}, {"dtype", dt.dtype}};
}

void from_json(const json& j, StructFieldInfo& dt) {
    dt.name = j.at("name").get<std::string>();
    dt.dtype = j.at("dtype").get<DataType>();
}

void to_json(json& j, const DataType& dt) {
    if (dt.typeCategory == DatatypeCategory::BUILTIN) {
        j = json{{"type", dt.typeCategory}, {"name", dt.name}};
    } else if (dt.typeCategory == DatatypeCategory::STRUCT) {
        j = json{{"type", dt.typeCategory}, {"fields", dt.fields}};
    } else if (dt.typeCategory == DatatypeCategory::TUPLE) {
        if (dt.tupleElementType == DatatypeCategory::STRUCT) {
            j = json{{"type", dt.typeCategory},
                     {"size", dt.tupleSize},
                     {"dtype", {{"type", dt.tupleElementType}, {"fields", dt.fields}}}};
        } else {
            j = json{{"type", dt.typeCategory},
                     {"size", dt.tupleSize},
                     {"dtype", json{{"name", dt.name}, {"type", dt.tupleElementType}}}};
        }
    } else {  // custom
        j = json{{"type", dt.typeCategory}, {"bytes", dt.bytes}, {"name", "none"}};
    }
}

void from_json(const json& j, DataType& dt) {
    auto dtCategory = j.at("type").get<std::string>();
    if (dtCategory == "builtin") {
        dt.typeCategory = DatatypeCategory::BUILTIN;
        j.at("name").get_to(dt.name);
    } else {
        dt.typeCategory = DatatypeCategory::CUSTOM;
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
            {"metadata", programMeta.kernels},
        };
    } else {
        j = json{
            {"dependencies", json::object()},  // INFO: always empty object, can't define the type
            {"metadata", programMeta.kernels},
        };
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
