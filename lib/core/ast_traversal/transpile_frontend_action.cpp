#include "core/ast_traversal/transpile_frontend_action.h"
#include "core/ast_traversal/transpile_ast_consumer.h"
#include "core/diag/diag_consumer.h"
#include "core/transpiler_session/session_stage.h"

#include <memory>

namespace oklt {

using namespace clang;

TranspileFrontendAction::TranspileFrontendAction(TranspilerSession& session)
    : _session(session),
      _stage(nullptr) {}

std::unique_ptr<ASTConsumer> TranspileFrontendAction::CreateASTConsumer(CompilerInstance& compiler,
                                                                        llvm::StringRef in_file) {
    _stage = std::make_unique<SessionStage>(_session, compiler);
    auto astConsumer = std::make_unique<TranspileASTConsumer>(*_stage);
    compiler.getDiagnostics().setClient(new DiagConsumer(*_stage));
    return std::move(astConsumer);
}

void TranspileFrontendAction::EndSourceFileAction() {
    assert(_stage != nullptr);
    //_session.output.kernel.sourceCode = _stage->getRewriterResult();
}
}  // namespace oklt
