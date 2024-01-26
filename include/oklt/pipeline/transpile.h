#pragma once

#include <oklt/core/config.h>
#include <oklt/pipeline/stages/transpiler/transpiler.h>

namespace oklt {

struct TranspileInput : public TranspileData {

  static tl::expected<TranspileInput, Error> build(const std::string &json);
  explicit TranspileInput(TRANSPILER_TYPE backend,
                          const std::string &sourceCode,
                          const std::filesystem::path &sourcePath,
                          const std::list<std::filesystem::path> &inlcudeDirectories,
                          const std::list<std::string> &defines);
  //TODO: add builder from Json, currently ctor can't throw
  TranspileData &getData();
  TRANSPILER_TYPE backend;
};

ExpectTranspilerResult transpile(TranspileInput input);
}
