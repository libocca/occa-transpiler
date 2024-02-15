#pragma once

#include <vector>
#include <string>
#include <optional>
#include <charconv>

namespace oklt::util {
std::string toLower(const std::string& str);
std::string toCamelCase(std::string str);
std::string pointerToStr(const void* ptr);

[[nodiscard]] std::string_view rtrim(std::string_view &str);
[[nodiscard]] std::string_view ltrim(std::string_view &str);
[[nodiscard]] std::string_view trim(std::string_view &str);
[[nodiscard]] std::string_view unParen(std::string_view& str);
[[nodiscard]] std::string_view slice(const std::string_view &str, size_t start, size_t end);
[[nodiscard]] std::vector<std::string_view> split(const std::string_view& str, const std::string_view& sep, int maxN = -1,
                                             bool keepEmpty = true);

template<typename T, bool>
std::optional<T> parseStrTo(std::string_view& str);

template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
inline std::optional<T> parseStrTo(std::string_view& str) {
    T ret;
    if (std::from_chars(str.data(), str.data() + str.size(), ret).ec == std::errc{})
        return ret;
    return {};
}

template<typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
inline std::optional<T> parseStrTo(std::string_view& str) {
    T ret;
    if (std::from_chars(str.data(), str.data() + str.size(), ret).ec == std::errc{})
        return ret;
    return {};
}

template<typename T>
inline std::optional<T> parseStrTo(const char *str) {
    std::string_view s(str);
    return parseStrTo<T>(s);
}

template<typename T>
inline std::optional<T> parseStrTo(const std::string &str) {
    std::string_view s(str);
    return parseStrTo<T>(s);
}

}  // namespace oklt::util
