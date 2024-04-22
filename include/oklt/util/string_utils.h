#pragma once

#include <oklt/core/error.h>
#include <charconv>
#include <optional>
#include <sstream>
#include <string>
#include <tl/expected.hpp>
#include <vector>

namespace oklt::util {
std::string toLower(const std::string& str);
std::string toCamelCase(std::string str);
std::string pointerToStr(const void* ptr);

[[nodiscard]] std::string_view rtrim(std::string_view& str);
[[nodiscard]] std::string_view ltrim(std::string_view& str);
[[nodiscard]] std::string_view trim(std::string_view& str);
[[nodiscard]] std::string_view unParen(std::string_view& str);
[[nodiscard]] std::string_view slice(const std::string_view& str, size_t start, size_t end);
[[nodiscard]] std::vector<std::string_view> split(const std::string_view& str,
                                                  const std::string_view& sep,
                                                  int maxN = -1,
                                                  bool keepEmpty = true);
std::string replace(std::string_view str, std::string_view from, std::string_view to);

template <typename T, bool>
std::optional<T> parseStrTo(std::string_view& str);

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
inline std::optional<T> parseStrTo(std::string_view& str) {
    T ret;
    if (std::from_chars(str.data(), str.data() + str.size(), ret).ec == std::errc{})
        return ret;
    return {};
}

template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
inline std::optional<T> parseStrTo(std::string_view& str) {
    T ret;
    if (std::from_chars(str.data(), str.data() + str.size(), ret).ec == std::errc{})
        return ret;
    return {};
}

template <typename T>
inline std::optional<T> parseStrTo(const char* str) {
    std::string_view s(str);
    return parseStrTo<T>(s);
}

template <typename T>
inline std::optional<T> parseStrTo(const std::string& str) {
    std::string_view s(str);
    return parseStrTo<T>(s);
}

namespace impl {
const std::string fmtBrackets = "{}";

tl::expected<size_t, Error> getCurlyBracketIdx(const std::string_view& str);

template <typename T>
std::string insertFmtValue(std::string str, size_t bracketIdx, const T& value) {
    // Stringify if possible
    std::stringstream ss;
    ss << value;
    return str.replace(bracketIdx, fmtBrackets.size(), ss.str());
}
}  // namespace impl

// Base case
tl::expected<std::string, Error> fmt(const std::string& s);

template <typename T, typename... Types>
tl::expected<std::string, Error> fmt(const std::string& s, T val, Types... vals) {
    auto pos = impl::getCurlyBracketIdx(s);
    if (!pos.has_value()) {
        return tl::make_unexpected(pos.error());
    }
    if (pos.value() == static_cast<size_t>(-1)) {
        return tl::make_unexpected(
            Error{std::error_code(), "fmt: Too much values (can't find '{}')"});
    }
    return fmt(impl::insertFmtValue(std::move(s), pos.value(), val), vals...);
}
}  // namespace oklt::util
