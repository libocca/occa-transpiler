#include "core/ast_traversal/transpile_ast_consumer.h"
#include "core/ast_processor_manager/ast_processor_manager.h"
#include "core/ast_traversal/preorder_traversal_nlr.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _stage(stage) {}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
    TranslationUnitDecl* tu = context.getTranslationUnitDecl();

    auto result =
        PreorderNlrTraversal(AstProcessorManager::instance(), _stage).applyAstProcessor(tu);
    if (!result) {
        return;
    }
}

SessionStage& TranspileASTConsumer::getSessionStage() {
    return _stage;
}

}  // namespace oklt
