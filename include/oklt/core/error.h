#pragma once

#include <string>
#include <system_error>
#include <any>

namespace oklt {

struct Error {
    std::error_code ec;
    std::string message;
    std::any ctx;
};

struct Warning {
    std::string desc;
};

}  // namespace oklt
