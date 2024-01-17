#include "oklt/core/transpile_session/error_reporter.h"

namespace oklt {
void ErrorReporter::emitFatalError(const clang::SourceRange &range,
                                   const std::string &what)
{}

void ErrorReporter::emitError(const clang::SourceRange &range, const std::string &what)
{}

void ErrorReporter::emitWarning(const clang::SourceRange &range, const std::string &what)
{}
}
