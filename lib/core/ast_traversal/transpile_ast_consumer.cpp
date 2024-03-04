#include "core/ast_traversal/transpile_ast_consumer.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_traversal/preorder_traversal_nlr.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _stage(stage) {}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
    TranslationUnitDecl* tu = context.getTranslationUnitDecl();

    auto result =
        PreorderNlrTraversal(AstProcessorManager::instance(), _stage).applyAstProccessor(tu);
    if (!result) {
        return;
    }

    _stage.getSession().output.kernel.sourceCode = std::move(result.value());
}

SessionStage& TranspileASTConsumer::getSessionStage() {
    return _stage;
}

}  // namespace oklt
