#pragma once

#include <map>
#include <string>

namespace oklt {
struct TransformedHeaders {
    // name to file content map
    std::map<std::string, std::string> fileMap;
};
}  // namespace oklt
