#pragma once

#include <clang/Basic/SourceLocation.h>
#include <string>

namespace oklt {

class ErrorReporter {
public:
  ErrorReporter() = default;
  ~ErrorReporter() = default;

  void emitFatalError(const clang::SourceRange &range, const std::string &what);
  void emitError(const clang::SourceRange &range, const std::string &what);
  void emitWarning(const clang::SourceRange &range, const std::string &what);
};
}
