#pragma once

#include <oklt/core/kernel_info/kernel_info.h>

#include <string>
#include <tl/expected.hpp>

namespace oklt {
struct UserOutput {
  struct {
    std::string sourceCode;
    std::string metadataJson;
  } normalized;

  struct {
    std::string sourceCode;
    std::string metadataJson;
    ProgramMetaData programMetaData;
  } kernel;

  struct {
    std::string sourceCode;
    std::string metadataJson;
  } launcher;
};

struct Error;
using UserResult = tl::expected<UserOutput, std::vector<Error>>;
}  // namespace oklt
