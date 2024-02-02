#pragma once

#include <oklt/core/target_backends.h>

#include <vector>

namespace clang {
class StoredDiagnostic;
}

namespace oklt {

struct Error;
struct Warning;
struct TranspilerSession;

using SharedTranspilerSession = std::shared_ptr<TranspilerSession>;

struct TranspilerSession {
  explicit TranspilerSession(TargetBackend backend, std::string sourceCode);

  struct UserInput {
    TargetBackend backend;
    std::string sourceCode;
    std::filesystem::path sourcePath;
    std::vector<std::filesystem::path> inlcudeDirectories;
    std::vector<std::string> defines;
  };

  explicit TranspilerSession(UserInput input);

  void pushDiagnosticMessage(clang::StoredDiagnostic& message);

  void pushError(std::error_code ec, std::string desc);
  const std::vector<Error>& getErrors() const;
  std::vector<Error>& getErrors();

  const std::vector<Warning>& getWarnings() const;
  std::vector<Warning>& getWarnings();

  struct UserOutput {
    struct {
      std::string outCode;
      std::string metadataJson;
    } normalized;

    struct {
      std::string outCode;
      std::string metadataJson;
    } kernel;

    struct {
      std::string outCode;
      std::string metadataJson;
    } launcher;
  };

  UserInput input;
  UserOutput output;
  // INFO: add fields here
 private:
  std::vector<Error> _errors;
  std::vector<Warning> _warnings;
};
}  // namespace oklt
