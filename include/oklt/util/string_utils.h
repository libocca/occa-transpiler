#pragma once

#include <string>
#include <sstream>

namespace oklt::util {
std::string toLower(const std::string& str);
std::string toCamelCase(std::string str);
std::string pointerToStr(const void* ptr);

template <typename InputIt>
std::string join(InputIt first, InputIt last, const std::string& separator = ", ") {
    std::stringstream ss;
    if (first == last) {
        return ss.str();
    }

    ss << *first;

    while (++first != last) {
        ss << separator << *first;
    }

    return ss.str();
}
}  // namespace oklt::util
