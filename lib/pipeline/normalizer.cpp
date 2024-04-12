#include <oklt/core/error.h>
#include <oklt/util/format.h>

#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_runner.h"
#include "pipeline/stages/normalizer/normalizer.h"

namespace oklt {
UserResult normalize(UserInput input) {
    return runNormalizerStage(TranspilerSession::make(std::move(input))).and_then(toUserResult);
}

UserResult normalize_ex(UserInput input) {
    static std::vector<std::string> normalizePipeline = {{OKL_DIRECTIVE_EXPANSION_STAGE},
                                                         {MACRO_EXPANSION_STAGE},
                                                         {OKL_TO_GNU_ATTR_NORMALIZER_STAGE},
                                                         {GNU_TO_STD_ATTR_NORMALIZER_STAGE}};

    auto session = TranspilerSession::make(std::move(input));
    auto result = runPipeline(normalizePipeline, session);
    return toUserResult(result.value());
}
}  // namespace oklt
