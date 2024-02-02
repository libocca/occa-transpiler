#include <oklt/core/error.h>
#include <oklt/pipeline/stages/normalizer/normalizer.h>

namespace oklt {
TranspilerSessionResult normalize(SharedTranspilerSession session) {
  return runNormalizerStage(session);
}
}  // namespace oklt
