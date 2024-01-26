#pragma once

#include "tl/expected.hpp"
#include <optional>
#include <filesystem>
#include <iosfwd>
#include <list>
#include <filesystem>

#include <oklt/core/transpiler_session/transpiler_session.h>

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

struct TranspileData {
  std::string sourceCode;
  std::filesystem::path sourcePath;
  std::list<std::filesystem::path> inlcudeDirectories;
  std::list<std::string> defines;
};

//TODO: needs definition
struct Error {
  //INFO: temporary solution to have somethign at least
  std::string desription;
};

using ExpectTranspilerResult = tl::expected<TranspilerResult, std::vector<Error>>;

ExpectTranspilerResult transpile(const TranspileData &input,
                                 TranspilerSession &session);

}

