#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/Basic/Diagnostic.h>

#include <clang/Frontend/TextDiagnostic.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

namespace oklt {

namespace {
clang::StoredDiagnostic substituteOriginalColumnIfNeeded(clang::StoredDiagnostic& diag,
                                                         SessionStage& stage) {
    auto& SM = stage.getCompiler().getSourceManager();
    auto& session = stage.getSession();
    clang::SourceLocation errLoc = diag.getLocation();
    auto fidOffset = SM.getDecomposedExpansionLoc(errLoc);
    if (session.attrOffsetToOriginalCol.find(fidOffset) != session.attrOffsetToOriginalCol.end()) {
        auto col = session.attrOffsetToOriginalCol[fidOffset];
        SPDLOG_DEBUG("Error col: {}, current offset: {}", col, fidOffset.second);

        // Create new loc with same line, but new col
        auto line = SM.getSpellingLineNumber(errLoc);
        errLoc = SM.translateLineCol(fidOffset.first, line, col);

        clang::StoredDiagnostic sd(diag.getLevel(),
                                   0,
                                   diag.getMessage(),
                                   clang::FullSourceLoc(errLoc, SM),
                                   clang::ArrayRef<clang::CharSourceRange>{},
                                   clang::ArrayRef<clang::FixItHint>{});
        return sd;
    } else if (session.attrOffsetToOriginalCol.find({fidOffset.first, fidOffset.second - 2}) !=
               session.attrOffsetToOriginalCol.end()) {
        auto col = session.attrOffsetToOriginalCol[{fidOffset.first, fidOffset.second - 2}];
        SPDLOG_DEBUG("Error col: {}, current offset: {}", col, fidOffset.second);

        // Create new loc with same line, but new col
        auto line = SM.getSpellingLineNumber(errLoc);
        errLoc = SM.translateLineCol(fidOffset.first, line, col);

        clang::StoredDiagnostic sd(diag.getLevel(),
                                   0,
                                   diag.getMessage(),
                                   clang::FullSourceLoc(errLoc, SM),
                                   clang::ArrayRef<clang::CharSourceRange>{},
                                   clang::ArrayRef<clang::FixItHint>{});
        return sd;

    } else {
        return diag;
    }
}
}  // namespace

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
    message = substituteOriginalColumnIfNeeded(message, stage);

    std::string formattedDiagnostics;
    llvm::raw_string_ostream os(formattedDiagnostics);
    clang::TextDiagnostic testDiagnostic(
        os, stage.getCompiler().getLangOpts(), &stage.getCompiler().getDiagnosticOpts());

    testDiagnostic.emitStoredDiagnostic(message);

    auto& originalLines = stage.getSession().originalLines;
    auto errLoc = message.getLocation();
    auto errRow = stage.getCompiler().getSourceManager().getSpellingLineNumber(errLoc);
    if (originalLines.find(errRow) != originalLines.end()) {
        auto originalLine = originalLines[errRow];
        auto lineMsgPrefix = fmt::format("{} | ", errRow);
        auto newLineMsg = lineMsgPrefix + originalLine;

        auto msgPrefixIdx = formattedDiagnostics.find(lineMsgPrefix);
        if (msgPrefixIdx != std::string::npos) {
            auto endIdx = formattedDiagnostics.find("\n", msgPrefixIdx);
            formattedDiagnostics.replace(msgPrefixIdx, endIdx - msgPrefixIdx, newLineMsg);
        }
    }

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
