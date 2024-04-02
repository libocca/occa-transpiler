#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/Basic/Diagnostic.h>

#include <clang/Frontend/TextDiagnostic.h>

namespace oklt {

SharedTranspilerSession TranspilerSession::make(UserInput input) {
    return std::make_shared<TranspilerSession>(std::move(input));
}

SharedTranspilerSession TranspilerSession::make(TargetBackend backend, std::string sourceCode) {
    return std::make_shared<TranspilerSession>(backend, sourceCode);
}

TranspilerSession::TranspilerSession(TargetBackend backend, std::string sourceCode) {
    input.backend = backend;
    input.sourceCode = std::move(sourceCode);
}

TranspilerSession::TranspilerSession(UserInput input_)
    : input(std::move(input_)) {}

void TranspilerSession::pushDiagnosticMessage(clang::StoredDiagnostic& message,
                                              SessionStage& stage) {
    std::string formattedDiagnostics;
    llvm::raw_string_ostream os(formattedDiagnostics);
    clang::TextDiagnostic testDiagnostic(
        os, stage.getCompiler().getLangOpts(), &stage.getCompiler().getDiagnosticOpts());

    testDiagnostic.emitStoredDiagnostic(message);

    //  create error category for syntax/semantic error/warning
    if (message.getLevel() > clang::DiagnosticsEngine::Level::Warning) {
        _errors.push_back(Error{std::error_code(), formattedDiagnostics});
    } else {
        _warnings.push_back(Warning{formattedDiagnostics});
    }
}

bool setCurrentKernelInfo(KernelInfo* ki);
[[nodiscard]] KernelInfo* getCurrentKernelInfo();
void TranspilerSession::pushError(std::error_code ec, std::string desc) {
    _errors.push_back(Error{ec, std::move(desc)});
}

void TranspilerSession::pushWarning(std::string desc) {
    _warnings.push_back(Warning{std::move(desc)});
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
