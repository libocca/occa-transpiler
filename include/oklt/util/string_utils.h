#pragma once

#include <string>

namespace oklt::util {
std::string toLower(const std::string& str);
std::string toCamelCase(std::string str);
std::string pointerToStr(const void* ptr);
}  // namespace oklt::util
