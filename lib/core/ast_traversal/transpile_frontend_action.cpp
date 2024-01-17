#include "oklt/core/ast_traversal/transpile_frontend_action.h"
#include "oklt/core/ast_traversal/transpile_ast_consumer.h"

namespace oklt {

using namespace clang;

TranspileFrontendAction::TranspileFrontendAction(TRANSPILER_TYPE backendType,
                                                 std::ostream &output)
    :_backend(backendType)
    , _output(output)
{}

std::unique_ptr<ASTConsumer> TranspileFrontendAction::CreateASTConsumer(
    CompilerInstance &compiler, llvm::StringRef in_file)
{
  return std::make_unique<TranspileASTConsumer>(
      TranspilerConfig {
        .backendType = _backend,
        .sourceFilePath = std::filesystem::path(in_file.str()),
        .transpiledOutput = _output
      },
      compiler.getASTContext());
}

}
