#pragma once

#include <string>
#include <system_error>

namespace oklt {

struct Error {
  std::error_code ec;
  std::string desc;
};

struct Warning {
  std::string desc;
};

}  // namespace oklt
