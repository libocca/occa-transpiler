#pragma once

#include <filesystem>
#include <tl/expected.hpp>

namespace oklt::util {
/**
 * @brief Reads a file and returns its content as a string.
 *
 * @param path The path to the file.
 * @return tl::expected<std::string, int> The content of the file if the reading was successful, or
 * an error code if the reading failed.
 */
tl::expected<std::string, int> readFileAsStr(const std::filesystem::path&);

/**
 * @brief Writes a string to a file.
 *
 * @param path The path to the file.
 * @param content The string to write to the file. (overwrites the file if it exists)
 * @return tl::expected<void, int> An empty expected object if the writing was successful, or an
 * error code if the writing failed.
 */
tl::expected<void, int> writeFileAsStr(const std::filesystem::path&, std::string_view);
}  // namespace oklt::util
