#include <oklt/core/error.h>
#include <oklt/util/format.h>

#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_runner.h"
#include "core/transpiler_session/session_result.h"

namespace oklt {
UserResult normalize(UserInput input) {
    static std::vector<std::string> normalizePipeline = {{OKL_DIRECTIVE_EXPANSION_STAGE},
                                                         {MACRO_EXPANSION_STAGE},
                                                         {OKL_TO_GNU_ATTR_NORMALIZER_STAGE},
                                                         {GNU_TO_STD_ATTR_NORMALIZER_STAGE}};

    auto session = TranspilerSession::make(std::move(input));
    auto result = runPipeline(normalizePipeline, session);
    if (!result) {
        return tl::make_unexpected(result.error());
    }
    return toUserResult(result.value());
}

}  // namespace oklt
