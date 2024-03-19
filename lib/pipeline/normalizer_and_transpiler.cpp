#include <oklt/core/error.h>
#include "pipeline/stages/normalizer/normalizer.h"
#include "pipeline/stages/transpiler/transpiler.h"

namespace oklt {
UserResult normalizeAndTranspile(UserInput input) {
    return runNormalizerStage(TranspilerSession::make(std::move(input)))
        .and_then(runTranspilerStage)
        .and_then(toUserResult);
}
}  // namespace oklt
