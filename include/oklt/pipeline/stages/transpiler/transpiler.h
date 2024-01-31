#pragma once

#include <filesystem>
#include <iosfwd>
#include <list>
#include <optional>
#include <tl/expected.hpp>

namespace oklt {

struct TranspilerSession;
struct Error;

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
