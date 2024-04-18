#include <oklt/core/error.h>

#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_runner.h"
#include "core/transpiler_session/session_result.h"

namespace oklt {
UserResult normalizeAndTranspile(UserInput input) {
    static std::vector<std::string> fullTranspilationPipeline = {{OKL_DIRECTIVE_EXPANSION_STAGE},
                                                                 {MACRO_EXPANSION_STAGE},
                                                                 {OKL_TO_GNU_ATTR_NORMALIZER_STAGE},
                                                                 {GNU_TO_STD_ATTR_NORMALIZER_STAGE},
                                                                 {TRANSPILATION_STAGE}};

    auto session = TranspilerSession::make(std::move(input));
    auto result = runPipeline(fullTranspilationPipeline, session);
    if (!result) {
        return tl::make_unexpected(result.error());
    }
    return toUserResult(result.value());
}
}  // namespace oklt
