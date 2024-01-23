#pragma once

#include "oklt/core/config.h"
#include "tl/expected.hpp"
#include <optional>
#include <filesystem>
#include <iosfwd>
#include <list>
#include <filesystem>


namespace oklt {
struct TranspilerResult {
  struct {
    std::string outCode;
    std::string metadataJson;
  } kernel;

  struct {
    std::string outCode;
    std::string metadataJson;
  }launcher;
};

struct TranspilerInput {
  //INFO: add
  // 1. metadata
  // 2. include directlries
  // 3. defines
  // 4. source path ???
  // 5. error handler ??
  std::string sourceCode;
  std::filesystem::path sourcePath;
  std::list<std::filesystem::path> inlcudeDirectories;
  std::list<std::string> defines;
  TRANSPILER_TYPE targetBackend;
  bool normalization;
};

//TODO: change error type
tl::expected<TranspilerInput, std::string> make_transpile_input(const std::filesystem::path &sourceFile,
                                                               const std::string &json);

//TODO: needs definition
struct Error {
  //INFO: temporary solution to have somethign at least
  std::string desription;
};

tl::expected<TranspilerResult,std::vector<Error>> transpile(TranspilerInput input);

}

