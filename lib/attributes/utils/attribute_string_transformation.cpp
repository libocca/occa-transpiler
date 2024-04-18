#include "attributes/utils/attribute_string_transformation.h"
#include <regex>

namespace oklt {
// [[okl_{attribute_name}(param-list)]] -> [[okl_{attribute_name}("(param-list)")]]
std::string wrapOklAttrParamsInString(const std::string src) {
    // Regular expression pattern

    std::regex pattern("\\[\\[okl_([^\\[\\]]+)(?:\\(([^\\)]*)\\))?\\]\\]");

    // Replacement string
    std::string replacement = "[[okl_$1(\"$2\")]]";

    // Perform substitution
    std::string dst = std::regex_replace(src, pattern, replacement);

    return dst;
}
}  // namespace oklt
