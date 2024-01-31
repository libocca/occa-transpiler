#pragma once

#include "oklt/core/config.h"
#include <string>

namespace oklt {

struct Error;

struct TranspilerSession {
  explicit TranspilerSession(TRANSPILER_TYPE backend);

  TRANSPILER_TYPE targetBackend;
  std::string transpiledCode;
  std::vector<Error> diagMessages;
  // INFO: add fields here
};
}  // namespace oklt
