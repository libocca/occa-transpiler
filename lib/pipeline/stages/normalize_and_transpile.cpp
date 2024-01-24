#include <oklt/pipeline/normalize_and_transpile.h>
#include <oklt/pipeline/stages/normalizer/normalizer.h>
#include <oklt/core/ast_traversal/transpile_frontend_action.h>
#include <oklt/core/transpiler_session/transpiler_session.h>
#include <clang/Tooling/Tooling.h>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;


namespace oklt {
tl::expected<TranspilerResult,std::vector<Error>> normalize_and_transpile(TranspilerInput input)
{
  TranspilerSession session {input.targetBackend};
  auto normalizerResult = normalize(NormalizerInput {input.sourceCode}, session);
  //TODO: needs unification of error interface or make cast
  if(!normalizerResult) {
    return tl::unexpected(std::vector<Error>{Error {"Normalized error"}});
  }

  Twine tool_name = "okl-transpiler";
  std::string rawFileName = input.sourcePath.filename().string();
  Twine file_name(rawFileName);
  std::vector<std::string> args = {
    "-std=c++17",
    "-fparse-all-comments",
    "-I."
  };

  Twine code(normalizerResult.value().cppSource);
  std::shared_ptr<PCHContainerOperations> pchOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<oklt::TranspileFrontendAction> action =
    std::make_unique<oklt::TranspileFrontendAction>(session);

  bool ret = runToolOnCodeWithArgs(std::move(action),
                                   code,
                                   args,
                                   file_name,
                                   tool_name,
                                   std::move(pchOps));
  if(!ret) {
    return tl::unexpected(std::vector<Error>{});
  }
  TranspilerResult result;
  result.kernel.outCode = session.transpiledCode;
  return result;
}
}
