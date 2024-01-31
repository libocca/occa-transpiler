#include <oklt/core/ast_traversal/transpile_frontend_action.h>
#include <oklt/core/transpiler_session/transpiler_session.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/diag/error.h>
#include <oklt/pipeline/stages/normalizer/normalizer.h>
#include <oklt/pipeline/stages/transpiler/transpiler.h>
#include <clang/Tooling/Tooling.h>
#include <llvm/Support/JSON.h>
#include <llvm/Support/raw_os_ostream.h>

#include <fstream>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {

ExpectTranspilerResult transpile(const TranspileData& input, TranspilerSession& session) {
  Twine tool_name = "okl-transpiler";
  std::string rawFileName = input.sourcePath.filename().string();
  Twine file_name(rawFileName);
  std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

  Twine code(input.sourceCode);
  std::shared_ptr<PCHContainerOperations> pchOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<oklt::TranspileFrontendAction> action =
    std::make_unique<oklt::TranspileFrontendAction>(session);

  bool ret =
    runToolOnCodeWithArgs(std::move(action), code, args, file_name, tool_name, std::move(pchOps));
  if (!ret) {
    return tl::unexpected(std::move(session.diagMessages));
  }
  TranspilerResult result;
  result.kernel.outCode = session.transpiledCode;
  return result;
}
}  // namespace oklt
