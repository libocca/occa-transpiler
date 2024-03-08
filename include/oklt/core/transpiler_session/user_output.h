#pragma once

#include <map>
#include <string>
#include <tl/expected.hpp>

namespace oklt {
struct UserOutput {
    struct {
        std::string sourceCode;
        std::map<std::string, std::string> sourceCodes;
        std::string metadataJson;
    } normalized;

    struct {
        std::string sourceCode;
        std::string metadataJson;
    } kernel;

    struct {
        std::string sourceCode;
        std::string metadataJson;
    } launcher;
};

struct Error;
using UserResult = tl::expected<UserOutput, std::vector<Error>>;
}  // namespace oklt
