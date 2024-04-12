#include <oklt/core/error.h>

#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_runner.h"
#include "pipeline/stages/transpiler/transpiler.h"

namespace oklt {

UserResult transpile(UserInput input) {
    return runTranspilerStage(TranspilerSession::make(std::move(input))).and_then(toUserResult);
}

UserResult transpile_ex(UserInput input) {
    static std::vector<std::string> fullTranspilationPipeline = {{OKL_DIRECTIVE_EXPANSION_STAGE},
                                                                 {MACRO_EXPANSION_STAGE},
                                                                 {OKL_TO_GNU_ATTR_NORMALIZER_STAGE},
                                                                 {GNU_TO_STD_ATTR_NORMALIZER_STAGE},
                                                                 {TRANSPILATION_STAGE}};

    auto session = TranspilerSession::make(std::move(input));
    auto result = runPipeline(fullTranspilationPipeline, session);
    return toUserResult(session);
}
}  // namespace oklt
