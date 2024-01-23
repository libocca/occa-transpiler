#include "oklt/core/transpile.h"
#include "oklt/core/ast_traversal/transpile_frontend_action.h"
#include <oklt/pipeline/stages/normalizer/normalizer.h>

#include <llvm/Support/raw_os_ostream.h>
#include <clang/Tooling/Tooling.h>

#include <fstream>

using namespace llvm;
using namespace clang;
using namespace clang::tooling;

namespace oklt {

tl::expected<TranspilerResult,std::vector<Error>> transpile(TranspilerInput input)
{
  Twine tool_name = "okl-transpiler";
  std::string rawFileName = input.sourcePath.filename().string();
  Twine file_name(rawFileName);
  std::vector<std::string> args = {
      "-std=c++17",
      "-fparse-all-comments",
      "-I."
  };

  std::string sourceCode;
  
  if(input.normalization) {
    TranspilerSession session{TRANSPILER_TYPE::CUDA};
    sourceCode = oklt::normalize({.oklSource = input.sourceCode}, session).value().cppSource;
  } else {
    sourceCode = input.sourceCode;
  }

  oklt::TranspilerSession session {input.targetBackend};

  Twine code(sourceCode);
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

