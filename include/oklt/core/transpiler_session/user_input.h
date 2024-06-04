#pragma once

#include <oklt/core/target_backends.h>

#include <filesystem>
#include <map>
#include <vector>

namespace oklt {

/**
 * @brief Represents the user input for transpilation, normalization or both
 */
struct UserInput {
    TargetBackend backend;                                  ///< The target backend.
    std::string source;                                     ///< The source code of OKL program.
    std::map<std::string, std::string> headers;             ///< The headers.
    std::filesystem::path sourcePath;                       ///< The path to the source file.
    std::vector<std::filesystem::path> includeDirectories;  ///< The include directories.
    std::vector<std::string> defines;                       ///< The defined macroses.
    std::string hash;                                       ///< OKL hash
    // TODO: change to std::vector
    std::vector<std::filesystem::path> userIntrinsics;  ///< OKL user external intrincis folder
};

}  // namespace oklt
