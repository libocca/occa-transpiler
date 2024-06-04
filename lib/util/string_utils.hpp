#pragma once

#include <oklt/core/error.h>
#include <charconv>
#include <optional>
#include <sstream>
#include <string>
#include <tl/expected.hpp>
#include <vector>

namespace oklt::util {

/**
 * @brief Convert a string to lowercase
 *
 * @param str Input string
 * @return std::string Lowercase string
 */
std::string toLower(const std::string& str);

/**
 * @brief check a string starts with prefix value
 *
 * @param str Input string
 * @param str Prefix string
 * @return bool Is started with provided prefix
 */
bool startsWith(const std::string& value, const std::string& prefix);

/**
 * @brief Convert a pointer to a string
 *
 * @param ptr Pointer to convert
 * @return std::string String representation of the pointer
 */
std::string pointerToStr(const void* ptr);

/**
 * @brief Trim the right side of a string
 *
 * @param str Input string
 * @return std::string_view Trimmed string
 */
[[nodiscard]] std::string_view rtrim(std::string_view& str);

/**
 * @brief Trims the left side of a string.
 *
 * @param str The input string.
 * @return std::string_view The trimmed string.
 */
[[nodiscard]] std::string_view ltrim(std::string_view& str);

/**
 * @brief Trims both sides of a string.
 *
 * @param str The input string.
 * @return std::string_view The trimmed string.
 */
[[nodiscard]] std::string_view trim(std::string_view& str);

/**
 * @brief Removes parentheses from the start and end of a string, if they exist.
 *
 * @param str The input string.
 * @return std::string_view The string with parentheses removed.
 */
[[nodiscard]] std::string_view unParen(std::string_view& str);

/**
 * @brief Slices a string between two indices.
 *
 * @param str The input string.
 * @param start The start index.
 * @param end The end index.
 * @return std::string_view The sliced string.
 */
[[nodiscard]] std::string_view slice(const std::string_view& str, size_t start, size_t end);

/**
 * @brief Splits a string by a separator.
 *
 * @param str The input string.
 * @param sep The separator.
 * @param maxN The maximum number of splits.
 * @param keepEmpty Whether to keep empty strings in the result.
 * @return std::vector<std::string_view> The split strings.
 */
[[nodiscard]] std::vector<std::string_view> split(const std::string_view& str,
                                                  const std::string_view& sep,
                                                  int maxN = -1,
                                                  bool keepEmpty = true);
/**
 * @brief Replaces all occurrences of a substring in a string with another substring.
 *
 * @param str The input string.
 * @param from The substring to replace.
 * @param to The substring to replace with.
 * @return std::string The string with all occurrences of 'from' replaced with 'to'.
 */
[[nodiscard]] std::string replace(std::string_view str, std::string_view from, std::string_view to);

/**
 * @brief Parses a string to a specified type.
 *
 * @tparam T The type to parse to.
 * @param str The string to parse.
 * @return std::optional<T> The parsed value if successful, or an empty optional if the parsing
 * failed.
 */
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

/**
 * @brief Gets the index of the first curly bracket in a string.
 *
 * @param str The input string.
 * @return tl::expected<size_t, Error> The index of the first curly bracket if found, -1 if not
 * found, or an error if the formatting is invalid.
 */
tl::expected<size_t, Error> getCurlyBracketIdx(const std::string_view& str);

/**
 * @brief Inserts a value into a string at a bracket index.
 *
 * @tparam T The type of the value.
 * @param str The input string.
 * @param bracketIdx The index to insert the value at.
 * @param value The value to insert.
 * @return std::string The string with the value inserted.
 */
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

/**
 * @brief Formats a string by replacing curly brackets with provided values.
 *
 * @tparam T The type of the first value.
 * @tparam Types The types of the remaining values.
 * @param s The input string.
 * @param val The first value to replace a curly bracket with.
 * @param vals The remaining values to replace curly brackets with.
 * @return tl::expected<std::string, Error> The formatted string if successful, or an error if the
 * formatting failed.
 */
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
