#include <oklt/core/error.h>
#include <oklt/pipeline/stages/normalizer/normalizer.h>
#include <oklt/pipeline/stages/transpiler/transpiler.h>

namespace oklt {
TranspilerSessionResult normalizeAndTranspile(SharedTranspilerSession session) {
  return runNormalizerStage(session).and_then(runTranspilerStage);
}
}  // namespace oklt
