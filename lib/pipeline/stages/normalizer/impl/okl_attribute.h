#pragma once

#include <clang/Basic/SourceLocation.h>

namespace oklt {
struct OklAttribute {
    std::string raw;
    std::string name;
    std::string params;
    std::vector<size_t> tok_indecies;
};

inline static std::string wrapAsSpecificGnuAttr(const OklAttribute& attr) {
    if (attr.params.empty()) {
        return "__attribute__((okl_" + attr.name + R"((")" + "" + "\")))";
    }

    return "__attribute__((okl_" + attr.name + R"((")" + attr.params + "\")))";
}

inline static std::string wrapAsSpecificCxxAttr(const OklAttribute& attr) {
    if (attr.params.empty()) {
        return "[[okl::" + attr.name + R"((")" + "" + "\")]]";
    }

    return "[[okl::" + attr.name + R"((")" + attr.params + "\")]]";
}

}  // namespace oklt
