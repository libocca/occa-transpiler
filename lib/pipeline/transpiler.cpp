#include "pipeline/stages/transpiler/transpiler.h"
#include <oklt/core/error.h>

namespace oklt {

UserResult transpile(UserInput input) {
    return runTranspilerStage(TranspilerSession::make(std::move(input))).and_then(toUserResult);
}
}  // namespace oklt
