#pragma once

#include <oklt/core/target_backends.h>
#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/original_source_mapper.h"

#include <clang/Rewrite/Core/DeltaTree.h>

#include <map>
#include <vector>

namespace clang {
class StoredDiagnostic;
}

namespace oklt {

struct Error;
struct Warning;
struct TranspilerSession;
class SessionStage;

using SharedTranspilerSession = std::shared_ptr<TranspilerSession>;

struct TranspilerSession {
    static SharedTranspilerSession make(UserInput);
    static SharedTranspilerSession make(TargetBackend backend, std::string sourceCode);

    explicit TranspilerSession(TargetBackend backend, std::string sourceCode);
    explicit TranspilerSession(UserInput input);

    void pushDiagnosticMessage(clang::StoredDiagnostic& message, SessionStage& stage);

    void pushError(std::error_code ec, std::string desc);
    void pushWarning(std::string desc);
    [[nodiscard]] const std::vector<Error>& getErrors() const;
    std::vector<Error>& getErrors();

    [[nodiscard]] const std::vector<Warning>& getWarnings() const;
    std::vector<Warning>& getWarnings();

    [[nodiscard]] OriginalSourceMapper& getOriginalSourceMapper();

    // TODO add methods for user input/output
    UserInput input;
    UserOutput output;

    TransformedFiles normalizedHeaders;
    TransformedFiles transpiledHeaders;

   private:
    std::vector<Error> _errors;
    std::vector<Warning> _warnings;
    OriginalSourceMapper _sourceMapper;
};
}  // namespace oklt
