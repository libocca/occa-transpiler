#pragma once

#include <any>
#include <string>
#include <system_error>

namespace oklt {

/**
 * @brief Represents an error with an error code, error description, and generic context.
 */
struct Error {
    std::error_code ec;  ///< The error code.
    std::string desc;    ///< The description of the error.
    std::any ctx;        ///< Any additional information about the error.`
};

/**
 * @brief Represents a warning with a description.
 */
struct Warning {
    std::string desc;  ///< The description of the warning.
};

}  // namespace oklt
