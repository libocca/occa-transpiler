#include "oklt/core/ast_traversal/transpile_frontend_action.h"
#include "oklt/core/ast_traversal/transpile_ast_consumer.h"
#include "oklt/core/diag/diag_consumer.h"

#include <memory>

namespace oklt {

using namespace clang;

TranspileFrontendAction::TranspileFrontendAction(TranspilerSession &session)
    : _session(session)
{}

std::unique_ptr<ASTConsumer> TranspileFrontendAction::CreateASTConsumer(
    CompilerInstance &compiler, llvm::StringRef in_file)
{
  _stage = std::make_unique<SessionStage>(_session, compiler);
  return std::make_unique<TranspileASTConsumer>(*_stage);
}

void TranspileFrontendAction::ExecuteAction() {
  if (!_stage)
    return;

  _diag = std::make_unique<DiagConsumer>(*_stage);

  DiagnosticsEngine &Diagnostics = getCompilerInstance().getDiagnostics();
  Diagnostics.setClient(_diag.get(), false);
}

}
