#include <oklt/core/error.h>

#include "core/transpiler_session/session_result.h"
#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_runner.h"

namespace oklt {
UserResult transpile(UserInput input) {
    static std::vector<std::string> justTranspilationPipeline = {{TRANSPILATION_STAGE}};

    auto session = TranspilerSession::make(std::move(input));
    auto result = runPipeline(justTranspilationPipeline, session);
    if (!result) {
        return tl::make_unexpected(std::move(result.error()));
    }
    return toUserResult(result.value());
}
}  // namespace oklt
