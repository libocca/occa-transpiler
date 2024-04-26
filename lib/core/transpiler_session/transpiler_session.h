#pragma once

#include <oklt/core/target_backends.h>
#include <oklt/core/transpiler_session/user_input.h>
#include <oklt/core/transpiler_session/user_output.h>

#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/original_source_mapper.h"

#include <clang/Rewrite/Core/DeltaTree.h>

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
    struct StagedFile {
        std::string source;  ///< Current source code (changes between stages)
        std::map<std::string, std::string> headers;  ///< Current headers (changes between stages)
    };

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

    const UserInput& getInput() const { return _input; }

    const UserOutput& getOutput() const { return _output; }

    UserOutput& getOutput() { return _output; }

    const std::string& getStagedSource() const { return _stagedFile.source; }
    std::string& getStagedSource() { return _stagedFile.source; }

    const std::map<std::string, std::string>& getStagedHeaders() const { return _stagedFile.headers; }
    std::map<std::string, std::string>& getStagedHeaders() { return _stagedFile.headers; }

    void updateSourceHeaders() {
        _stagedFile.source = _output.normalized.source;
        _stagedFile.headers = _output.normalized.headers;
    }

   private:
    // TODO add methods for user input/output
    const UserInput _input;
    UserOutput _output;

    StagedFile _stagedFile;

    std::vector<Error> _errors;
    std::vector<Warning> _warnings;
    OriginalSourceMapper _sourceMapper;
};
}  // namespace oklt
