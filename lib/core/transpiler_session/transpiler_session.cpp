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

/**
 * @brief Find the first non-whitespace character in a string.
 * @param str The string to search.
 * @return An iterator pointing to the first non-whitespace character, or str.cend() if none found.
 */
std::string::const_iterator firstNonWhiteCharacter(const std::string& str) {
    if (str.empty()) {
        return str.cend();
    }
    auto it = str.cbegin();
    while (isspace(*it) && (it != str.cend())) {
        ++it;
    }
    return it;
}

/**
 * @brief Substitute the original column in a diagnostic message if needed. It is needed in case of
 * normalizer modified original source line.
 * @param diag The diagnostic message.
 * @param stage The current session stage.
 * @return A diagnostic message with the column substituted if needed.
 */
clang::StoredDiagnostic substituteOriginalColumnIfNeeded(clang::StoredDiagnostic& diag,
                                                         SessionStage& stage) {
    auto& sm = stage.getCompiler().getSourceManager();
    auto& session = stage.getSession();
    clang::SourceLocation errLoc = diag.getLocation();
    auto line = sm.getSpellingLineNumber(errLoc);
    auto [fid, offset] = sm.getDecomposedExpansionLoc(errLoc);
    auto& attrOffsetToOriginalCol = session.getOriginalSourceMapper().getAttrOffsetToOriginalCol();

    auto fidOffset = std::make_pair(fid, offset);
    if (attrOffsetToOriginalCol.find(fidOffset) != attrOffsetToOriginalCol.end()) {
        auto col = attrOffsetToOriginalCol.at(fidOffset);
        SPDLOG_DEBUG(
            "Substitute error for attribute at offset {} column to: {}", fidOffset.second, col);

        // Create new loc with same line, but new col
        errLoc = sm.translateLineCol(fidOffset.first, line, col);

        clang::StoredDiagnostic sd(
            diag.getLevel(), 0, diag.getMessage(), clang::FullSourceLoc(errLoc, sm), {}, {});
        return sd;
    }
    auto& originalLines = stage.getSession().getOriginalSourceMapper().getOriginalLines();
    auto errRow = stage.getCompiler().getSourceManager().getSpellingLineNumber(errLoc);

    // Otherwise, if there was a normalization, move column to first non-white char. symbol
    // TODO: Reuse deltatree, to actually show correct column
    if (originalLines.count({fid, errRow})) {
        auto origLine = originalLines.at({fid, errRow});
        auto col = std::distance(origLine.cbegin(), firstNonWhiteCharacter(origLine)) + 1;
        errLoc = sm.translateLineCol(fidOffset.first, line, col);
        clang::StoredDiagnostic sd(
            diag.getLevel(), 0, diag.getMessage(), clang::FullSourceLoc(errLoc, sm), {}, {});
        return sd;
    }

    return diag;
}
/**
 * @brief Removes the 'okl_' prefix from the attribute name in an error message.
 * @param input The input string.
 * @return The input string with the 'okl_' prefix removed before attribute name
 */
std::string removeOklPrefix(const std::string& input) {
    // match word 'okl_' prefix proceeding 'attribute' word
    std::regex pattern(fmt::format(R"('{}(.+?)' attribute)", OKL_ATTR_PREFIX));
    // Replacement string
    std::string replacement = "'$1' attribute";
    // Perform the replacement
    std::string result = std::regex_replace(input, pattern, replacement);

    return result;
}

/**
 * @brief Substitutes the original line for a given row in the error message. Substitution is done
 * only if normalization modified the original line.
 * @param input The input string -- error message.
 * @param ol The original lines.
 * @param fid The file ID.
 * @param row The row number.
 * @return The input string with the original line substituted for the given row.
 */
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

/**
 * @brief Substitutes the original line in a diagnostic message if needed (needed in case of
 * normalization).
 * @param diag The diagnostic message.
 * @param stage The current session stage.
 * @return A diagnostic message with the original line substituted if needed.
 */
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

/**
 * @brief Retrieves the error message from a diagnostic.
 * @param diag Stored diagnostic
 * @param stage The current session stage.
 * @return The error message.
 */
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
    _input.backend = backend;
    _input.source = std::move(sourceCode);
}

TranspilerSession::TranspilerSession(UserInput input)
    : _input(std::move(input)) {}

void TranspilerSession::pushDiagnosticMessage(clang::StoredDiagnostic& diag, SessionStage& stage) {
    auto errorMsg = getErrorMessage(diag, stage);
    //  create error category for syntax/semantic error/warning
    if (diag.getLevel() > clang::DiagnosticsEngine::Level::Warning) {
        _errors.push_back(Error{std::error_code(), errorMsg});
    } else {
        _warnings.push_back(Warning{errorMsg});
    }
}

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
