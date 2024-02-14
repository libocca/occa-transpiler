#include <oklt/core/ast_traversal/transpile_frontend_action.h>
#include <oklt/core/error.h>
#include <oklt/core/transpiler_session/session_result.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/transpiler_session/transpiler_session.h>

#include <clang/Tooling/Tooling.h>

#include <llvm/Support/raw_os_ostream.h>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {

TranspilerSessionResult runTranspilerStage(SharedTranspilerSession session) {
  auto& input = session->input;

  Twine tool_name = "okl-transpiler";
  std::string rawFileName = input.sourcePath.filename().string();
  Twine file_name(rawFileName);
  std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

  Twine code(input.sourceCode);
  std::shared_ptr<PCHContainerOperations> pchOps = std::make_shared<PCHContainerOperations>();
  std::unique_ptr<oklt::TranspileFrontendAction> action =
    std::make_unique<oklt::TranspileFrontendAction>(*session);

  bool ret =
    runToolOnCodeWithArgs(std::move(action), code, args, file_name, tool_name, std::move(pchOps));
  if (!ret) {
    return tl::make_unexpected(std::move(session->getErrors()));
  }

    return session;
}
}  // namespace oklt
