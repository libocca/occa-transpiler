#include <oklt/pipeline/normalize.h>

namespace oklt {
ExpecteNormalizerResult normalize(NormalizerInput input) {
  TranspilerSession session {TRANSPILER_TYPE::CUDA};
  return normalize(input, session);
}
}
