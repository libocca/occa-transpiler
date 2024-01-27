#pragma once

#include <filesystem>
#include <iosfwd>
#include <list>
#include <optional>
#include "tl/expected.hpp"

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
  } launcher;
};

struct TranspileData {
  std::string sourceCode;
  std::filesystem::path sourcePath;
  std::list<std::filesystem::path> inlcudeDirectories;
  std::list<std::string> defines;
};

using ExpectTranspilerResult = tl::expected<TranspilerResult, std::vector<Error>>;

ExpectTranspilerResult transpile(const TranspileData& input, TranspilerSession& session);

}  // namespace oklt
