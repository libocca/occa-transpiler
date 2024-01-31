#include <clang/Tooling/Tooling.h>
#include <oklt/core/ast_traversal/transpile_frontend_action.h>
#include <oklt/core/transpiler_session/transpiler_session.h>
#include <oklt/pipeline/normalize_and_transpile.h>
#include <oklt/pipeline/stages/normalizer/normalizer.h>
#include <oklt/core/diag/error.h>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {
ExpectTranspilerResult normalize_and_transpile(TranspileInput input) {
  TranspilerSession session{input.backend};
  auto normalizerResult = normalize(NormalizerInput{input.sourceCode}, session);
  // TODO: needs unification of error interface or make cast
  if (!normalizerResult) {
    return tl::unexpected(std::vector<Error>{Error{"Normalized error"}});
  }

  input.sourceCode = normalizerResult.value().cppSource;
  return transpile(input.getData(), session);
}
}  // namespace oklt
