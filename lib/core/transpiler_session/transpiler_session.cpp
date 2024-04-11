#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include "attributes/attribute_names.h"

#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/Basic/Diagnostic.h>

#include <clang/Frontend/TextDiagnostic.h>

#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <regex>

namespace {
using namespace oklt;
clang::StoredDiagnostic substituteOriginalColumnIfNeeded(clang::StoredDiagnostic& diag,
                                                         SessionStage& stage) {
    auto& sm = stage.getCompiler().getSourceManager();
    auto& session = stage.getSession();
    clang::SourceLocation errLoc = diag.getLocation();
    auto [fid, offset] = sm.getDecomposedExpansionLoc(errLoc);
    auto& attrOffsetToOriginalCol = session.getOriginalSourceMapper().getAttrOffsetToOriginalCol();

    auto fidOffset = std::make_pair(fid, offset);
    if (attrOffsetToOriginalCol.find(fidOffset) != attrOffsetToOriginalCol.end()) {
        auto col = attrOffsetToOriginalCol.at(fidOffset);
        SPDLOG_DEBUG(
            "Substitute error for attribute at offset {} column to: {}", fidOffset.second, col);

        // Create new loc with same line, but new col
        auto line = sm.getSpellingLineNumber(errLoc);
        errLoc = sm.translateLineCol(fidOffset.first, line, col);

        clang::StoredDiagnostic sd(
            diag.getLevel(), 0, diag.getMessage(), clang::FullSourceLoc(errLoc, sm), {}, {});
        return sd;
    }

    return diag;
}
std::string removeOklPrefix(const std::string& input) {
    // match word 'okl_' prefix proceeding 'attribute' word
    std::regex pattern(fmt::format(R"('{}(.+?)' attribute)", OKL_ATTR_PREFIX));
    // Replacement string
    std::string replacement = "'$1' attribute";
    // Perform the replacement
    std::string result = std::regex_replace(input, pattern, replacement);

    return result;
}

std::string& substituteOriginalLineForRow(std::string& input,
                                          const OriginalLines& ol,
                                          clang::FileID fid,
                                          unsigned row) {
    auto errFidRow = std::make_pair(fid, row);

    // Substitute original line if possible
    if (ol.find(errFidRow) == ol.end()) {
        return input;
    }

    auto originalLine = ol.at(errFidRow);
    auto lineMsgPrefix = fmt::format("{} | ", row);
    auto newLineMsg = lineMsgPrefix + originalLine;

    auto msgPrefixIdx = input.find(lineMsgPrefix);
    if (msgPrefixIdx == std::string::npos) {
        return input;
    }

    auto endIdx = input.find("\n", msgPrefixIdx);
    input.replace(msgPrefixIdx, endIdx - msgPrefixIdx, newLineMsg);
    SPDLOG_DEBUG("Substitute original line for row: {}", row);

    return input;
}
}  // namespace

namespace oklt {
std::string substituteOriginalLineIfNeeded(clang::StoredDiagnostic& diag,
                                           std::string errorMsg,
                                           SessionStage& stage) {
    auto& originalLines = stage.getSession().getOriginalSourceMapper().getOriginalLines();
    auto errLoc = diag.getLocation();
    auto errFid = stage.getCompiler().getSourceManager().getFileID(errLoc);
    auto errRow = stage.getCompiler().getSourceManager().getSpellingLineNumber(errLoc);
    auto errFidRow = std::make_pair(errFid, errRow);

    // remove OKL prefix for attribute error message
    errorMsg = removeOklPrefix(errorMsg);

    // Substitute original line if possible
    errorMsg = substituteOriginalLineForRow(errorMsg, originalLines, errFid, errRow);

    return errorMsg;
}

std::string getErrorMessage(clang::StoredDiagnostic& diag, SessionStage& stage) {
    diag = substituteOriginalColumnIfNeeded(diag, stage);

    std::string errorMsg;
    llvm::raw_string_ostream os(errorMsg);
    clang::TextDiagnostic testDiagnostic(
        os, stage.getCompiler().getLangOpts(), &stage.getCompiler().getDiagnosticOpts());

    testDiagnostic.emitStoredDiagnostic(diag);
    errorMsg = substituteOriginalLineIfNeeded(diag, errorMsg, stage);
    return errorMsg;
}

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
    auto errorMsg = getErrorMessage(diag, stage);
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
