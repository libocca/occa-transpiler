#pragma once

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
    //if (attr.params.empty()) {
    //    return "__attribute__((" + OKL_ATTR_PREFIX + attr.name + R"((")" + "" + "\")))";
    //}

    //return "__attribute__((" + OKL_ATTR_PREFIX + attr.name + R"((")" + attr.params + "\")))";
    return fmt::format("__attribute__(({}{}(\"{}\")))", OKL_ATTR_PREFIX, attr.name, attr.params);
}

inline static std::string wrapAsSpecificCxxAttr(const OklAttribute& attr) {
    //if (attr.params.empty()) {
    //    return "[[" + OKL_ATTR_PREFIX + attr.name + R"((")" + "" + "\")]]";
    //}

    //return "[[" + OKL_ATTR_PREFIX + attr.name + R"((")" + attr.params + "\")]]";
    return fmt::format("[[{}{}(\"{}\")]]", OKL_ATTR_PREFIX, attr.name, attr.params);
}

}  // namespace oklt
