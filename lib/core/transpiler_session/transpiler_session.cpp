#include "oklt/core/transpiler_session/transpiler_session.h"
#include "oklt/core/diag/error.h"

namespace oklt {

TranspilerSession::TranspilerSession(TRANSPILER_TYPE backend)
    : targetBackend(backend), transpiledCode() {}

}  // namespace oklt
