#pragma once

#include <map>
#include <string>

#include <tl/expected.hpp>

namespace oklt {
/**
 * @brief Represents the output of transpilation or nortmalization.
 */
struct UserOutput {
    struct {
        std::string source;                          ///< The normalized source code.
        std::map<std::string, std::string> headers;  ///< The normalized headers (relative path of
                                                     ///< header -> normalized source code)
    } normalized;

    struct {
        std::string source;    ///< The kernel source code.
        std::string metadata;  ///< The kernel metadata (dumped as JSON)
    } kernel;

    struct {
        std::string source;    ///< The launcher source code.
        std::string metadata;  ///< The launcher metadata (dumped as JSON)
    } launcher;
};

struct Error;
using UserResult = tl::expected<UserOutput, std::vector<Error>>;
}  // namespace oklt
