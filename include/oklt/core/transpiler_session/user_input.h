#pragma once

#include <oklt/core/target_backends.h>
#include <vector>

namespace oklt {

struct UserInput {
  TargetBackend backend;
  std::string sourceCode;
  std::filesystem::path sourcePath;
  std::vector<std::filesystem::path> inlcudeDirectories;
  std::vector<std::string> defines;
};

}  // namespace oklt
