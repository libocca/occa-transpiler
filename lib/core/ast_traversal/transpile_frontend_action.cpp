#include "oklt/core/ast_traversal/transpile_frontend_action.h"
#include "oklt/core/ast_traversal/transpile_ast_consumer.h"

namespace oklt {

using namespace clang;

TranspileFrontendAction::TranspileFrontendAction(TranspilerSession &session)
    : _session(session)
{}

std::unique_ptr<ASTConsumer> TranspileFrontendAction::CreateASTConsumer(
    CompilerInstance &compiler, llvm::StringRef in_file)
{
  return std::make_unique<TranspileASTConsumer>(
      SessionStage { _session, compiler});
}

}
