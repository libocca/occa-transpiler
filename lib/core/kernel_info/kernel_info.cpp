#include <oklt/core/kernel_info/kernel_info.h>

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

void to_json(json& j, const ParsedKernelInfo& kernelMeta) {
    j = json{{"arguments", kernelMeta.arguments}, {"name", kernelMeta.name}};
}

void from_json(const json& j, ParsedKernelInfo& kernelMeta) {
    j.at("arguments").get_to(kernelMeta.arguments);
    j.at("name").get_to(kernelMeta.name);
}

void to_json(json& j, const KernelMetadata& kernelInfo) {
    if (kernelInfo.props.has_value()) {
        j = json{
            {"dependencies", json::object()},  // INFO: always empty object, can't define the type
            {"hash", kernelInfo.hash},
            {"metadata", kernelInfo.metadata},
            {"props", kernelInfo.props.value()}};
    } else {
        j = json{
            {"dependencies", json::object()},  // INFO: always empty object, can't define the type
            {"hash", kernelInfo.hash},
            {"metadata", kernelInfo.metadata},
            {"props", json::object()}};
    }
}

void from_json(const json& j, KernelMetadata& kernelInfo) {
    kernelInfo.dependencies = std::nullopt;
    const auto& value = j.at("props");
    if (value.is_object() && !value.empty()) {
        j.at("hash").get_to(kernelInfo.hash);
        j.at("metadata").get_to(kernelInfo.metadata);
        PropertyInfo prop;
        value.get_to(prop);
        kernelInfo.props = prop;
    } else {
        j.at("hash").get_to(kernelInfo.hash);
        j.at("metadata").get_to(kernelInfo.metadata);
        kernelInfo.props = std::nullopt;
    }
}
}  // namespace oklt
