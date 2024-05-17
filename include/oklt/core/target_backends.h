#pragma once

#include <string>

#include "tl/expected.hpp"

namespace oklt {
/**
 * @brief Enum for the target backends.
 */
enum struct TargetBackend : unsigned char {
    SERIAL,  ///< Serial backend.
    OPENMP,  ///< OpenMP backend.
    CUDA,    ///< CUDA backend.
    HIP,     ///< HIP backend.
    DPCPP,   ///< DPCPP backend.
    METAL,   ///< Metal backend.

    _LAUNCHER,  ///< Launcher backend.
};

/**
 * @brief Converts a string to a TargetBackend.
 *
 * @param type The string to convert.
 * @return tl::expected<TargetBackend, std::string> The TargetBackend if the conversion was
 * successful, or an error message if the conversion failed.
 */
tl::expected<TargetBackend, std::string> backendFromString(const std::string& type);

/**
 * @brief Converts a TargetBackend to a string.
 *
 * @param backend The TargetBackend to convert.
 * @return std::string The string representation of the TargetBackend.
 */
std::string backendToString(TargetBackend backend);

/**
 * @brief Checks if a TargetBackend is a host backend (runs on the CPU).
 *
 * @param backend The TargetBackend to check.
 * @return bool True if the TargetBackend is a host category, false otherwise.
 */
bool isHostCategory(TargetBackend backend);

/**
 * @brief Checks if a TargetBackend is a device backend (runs on the GPU or other co-processor)
 *
 * @param backend The TargetBackend to check.
 * @return bool True if the TargetBackend is a device category, false otherwise.
 */
bool isDeviceCategory(TargetBackend backend);

}  // namespace oklt
