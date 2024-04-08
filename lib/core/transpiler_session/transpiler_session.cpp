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
    auto [fid, offset] = SM.getDecomposedExpansionLoc(errLoc);
    auto& attrOffsetToOriginalCol = session.getOriginalSourceMapper().getAttrOffsetToOriginalCol();

    // FIXME: Temporary hack to catch errors from frontend: location is pointing at attribute name,
    // not start (len("[[") == 2)
    std::array<uint32_t, 2> offsets{offset, offset - 2};
    for (const auto& currOffset : offsets) {
        auto fidOffset = std::make_pair(fid, currOffset);
        if (attrOffsetToOriginalCol.find(fidOffset) != attrOffsetToOriginalCol.end()) {
            auto col = attrOffsetToOriginalCol.at(fidOffset);
            SPDLOG_DEBUG("Error col: {}, current offset: {}", col, fidOffset.second);

            // Create new loc with same line, but new col
            auto line = SM.getSpellingLineNumber(errLoc);
            errLoc = SM.translateLineCol(fidOffset.first, line, col);

            clang::StoredDiagnostic sd(
                diag.getLevel(), 0, diag.getMessage(), clang::FullSourceLoc(errLoc, SM), {}, {});
            return sd;
        }
    }

    return diag;
}

std::string substituteOriginalLineIfNeeded(clang::StoredDiagnostic& diag,
                                           std::string errorMsg,
                                           SessionStage& stage) {
    auto& originalLines = stage.getSession().getOriginalSourceMapper().getOriginalLines();
    auto errLoc = diag.getLocation();
    auto errRow = stage.getCompiler().getSourceManager().getSpellingLineNumber(errLoc);
    // Substitute original line if possible
    if (originalLines.find(errRow) != originalLines.end()) {
        auto originalLine = originalLines.at(errRow);
        auto lineMsgPrefix = fmt::format("{} | ", errRow);
        auto newLineMsg = lineMsgPrefix + originalLine;

        auto msgPrefixIdx = errorMsg.find(lineMsgPrefix);
        if (msgPrefixIdx != std::string::npos) {
            auto endIdx = errorMsg.find("\n", msgPrefixIdx);
            errorMsg.replace(msgPrefixIdx, endIdx - msgPrefixIdx, newLineMsg);
        }
    }
    return errorMsg;
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

void TranspilerSession::pushDiagnosticMessage(clang::StoredDiagnostic& diag, SessionStage& stage) {
    diag = substituteOriginalColumnIfNeeded(diag, stage);

    std::string errorMsg;
    llvm::raw_string_ostream os(errorMsg);
    clang::TextDiagnostic testDiagnostic(
        os, stage.getCompiler().getLangOpts(), &stage.getCompiler().getDiagnosticOpts());

    testDiagnostic.emitStoredDiagnostic(diag);
    errorMsg = substituteOriginalLineIfNeeded(diag, errorMsg, stage);

    //  create error category for syntax/semantic error/warning
    if (diag.getLevel() > clang::DiagnosticsEngine::Level::Warning) {
        _errors.push_back(Error{std::error_code(), errorMsg});
    } else {
        _warnings.push_back(Warning{errorMsg});
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

OriginalSourceMapper& TranspilerSession::getOriginalSourceMapper() {
    return _sourceMapper;
}
}  // namespace oklt
