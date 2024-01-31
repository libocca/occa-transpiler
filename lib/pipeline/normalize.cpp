#include <oklt/pipeline/normalize.h>
#include <oklt/core/transpiler_session/transpiler_session.h>
#include <oklt/core/diag/error.h>

namespace oklt {
ExpecteNormalizerResult normalize(NormalizerInput input) {
  TranspilerSession session{TRANSPILER_TYPE::CUDA};
  return normalize(input, session);
}
}  // namespace oklt
