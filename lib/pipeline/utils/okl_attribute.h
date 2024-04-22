#pragma once

#include <oklt/util/string_utils.h>
#include "attributes/attribute_names.h"

#include <spdlog/fmt/fmt.h>

#include <vector>

namespace oklt {

struct OklAttribute {
    std::string raw;
    std::string name;
    std::string params;
    std::vector<size_t> tok_indecies;
};

inline static std::string wrapAsSpecificGnuAttr(const OklAttribute& attr) {
    return fmt::format(
        "__attribute__(({}{}(\"{}\")))", OKL_ATTR_PREFIX, attr.name, attr.params);
}

inline static std::string wrapAsSpecificCxxAttr(const OklAttribute& attr) {
    // TODO: this is ugly
    auto paramsWithEscape = util::replace(attr.params, "\"", "\\\"");
    auto res = fmt::format("[[{}{}(\"{}\")]]", OKL_ATTR_PREFIX, attr.name, paramsWithEscape);
    return res;
}

}  // namespace oklt
