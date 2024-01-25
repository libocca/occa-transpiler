#include <oklt/pipeline/transpile.h>


namespace oklt {

tl::expected<TranspileInput, Error> build(const std::string &json) {
  return tl::unexpected<Error>(Error {"not implemented"});
}

TranspileInput::TranspileInput(TRANSPILER_TYPE backend,
                        const std::string &sourceCode,
                        const std::filesystem::path &sourcePath,
                        const std::list<std::filesystem::path> &inlcudeDirectories,
                        const std::list<std::string> &defines)
    :TranspileData {
      .sourceCode = sourceCode,
      .sourcePath = sourcePath,
      .inlcudeDirectories = inlcudeDirectories,
      .defines = defines
    }
    , backend(backend)
{}

TranspileData &TranspileInput::getData() {
  return *this;
}


ExpectTranspilerResult transpile(TranspileInput input) {
  TranspilerSession session {input.backend};
  return transpile(input.getData(), session);
}
}
