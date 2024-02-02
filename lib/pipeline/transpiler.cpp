#include <oklt/core/error.h>
#include <oklt/pipeline/stages/transpiler/transpiler.h>

namespace oklt {

TranspilerSessionResult transpile(SharedTranspilerSession session) {
  return runTranspilerStage(session);
}
}  // namespace oklt
