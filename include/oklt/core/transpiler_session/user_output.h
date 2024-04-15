#pragma once

#include <map>
#include <string>

#include <tl/expected.hpp>

namespace oklt {
struct UserOutput {
    struct {
        std::string source;
        std::map<std::string, std::string> headers;
    } normalized;

    struct {
        std::string source;
        std::string metadata;
    } kernel;

    struct {
        std::string source;
        std::string metadata;
    } launcher;
};

struct Error;
using UserResult = tl::expected<UserOutput, std::vector<Error>>;
}  // namespace oklt
