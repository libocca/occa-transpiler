#include "attributes/utils/parse.h"

namespace oklt {
using namespace clang;

// Given C++ suppress attribute returns it's parameter as string. Expected attribute to have only
// one attribute
tl::expected<std::string, Error> parseCppAttributeParameterStr(const clang::SuppressAttr* attr) {
    size_t nParams = 0;
    for (const auto& param : attr->diagnosticIdentifiers()) {
        ++nParams;
    }
    if (nParams != 1) {
        return tl::make_unexpected(
            Error{std::error_code(), "'CPPfied' OKL attributes must have only one argument"});
    }

    auto param = attr->diagnosticIdentifiers().begin();
    std::string param_str(param->data(), param->size());
    //   Remove '(' and ')' and the beginning and end
    param_str = param_str.substr(1, param_str.size() - 2);

    return param_str;
}

namespace {
std::vector<std::string> split(const std::string& str, const std::string& delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = str.find(delimiter, pos_start)) != std::string::npos) {
        token = str.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(str.substr(pos_start));
    return res;
}

}  // namespace

std::vector<std::string> splitCSV(const std::string& str) {
    return split(str, ",");
}

tl::expected<std::vector<std::string>, Error> parseOKLAttributeParamsStr(const clang::Attr* a) {
    // TODO: add support for annotate attribute
    auto supressAttr = dyn_cast_or_null<SuppressAttr>(a);
    if (supressAttr) {
        auto paramStr = parseCppAttributeParameterStr(supressAttr);
        if (!paramStr.has_value()) {
            return tl::make_unexpected(paramStr.error());
        }
        auto paramStrs = splitCSV(paramStr.value());
        return paramStrs;
    }
    return tl::make_unexpected(
        Error{std::error_code(), "OKL attribute must be a supress cpp attribute"});
}
}  // namespace oklt
