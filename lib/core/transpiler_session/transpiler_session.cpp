#include <oklt/core/error.h>
#include <oklt/core/transpiler_session/transpiler_session.h>

#include <clang/Basic/Diagnostic.h>

namespace oklt {

TranspilerSession::TranspilerSession(TargetBackend backend, std::string sourceCode) {
  input.backend = backend;
  input.sourceCode = std::move(sourceCode);
}

TranspilerSession::TranspilerSession(TranspilerSession::UserInput input_)
    : input(std::move(input_)) {}

void TranspilerSession::pushDiagnosticMessage(clang::StoredDiagnostic& message) {
  // TODO: Fixup sourceLocation
  auto msg = message.getMessage();
  auto lineNo = message.getLocation().getLineNumber();

  std::stringstream ss;
  ss << "line " << lineNo << ": ";
  ss << msg.str();
  // TODO
  //  create error category for syntax/semantic error/warning
  if (message.getLevel() > clang::DiagnosticsEngine::Level::Warning) {
    _errors.push_back(Error{std::error_code(), ss.str()});
  } else {
    _warnings.push_back(Warning{ss.str()});
  }
}

void TranspilerSession::pushError(std::error_code ec, std::string desc) {
  _errors.push_back(Error{ec, std::move(desc)});
}

const std::vector<Error>& TranspilerSession::getErrors() const {
  return _errors;
}

std::vector<Error>& TranspilerSession::getErrors() {
  return _errors;
}

const std::vector<Warning>& TranspilerSession::getWarnings() const {
  return _warnings;
}

std::vector<Warning>& TranspilerSession::getWarnings() {
  return _warnings;
}
}  // namespace oklt
