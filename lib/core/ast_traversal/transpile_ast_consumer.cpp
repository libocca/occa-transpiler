#include <oklt/core/ast_processor_manager/ast_processor_manager.h>
#include <oklt/core/ast_traversal/preorder_traversal_nlr.h>
#include <oklt/core/ast_traversal/transpile_ast_consumer.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _stage(stage) {}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
    TranslationUnitDecl* tu = context.getTranslationUnitDecl();

    PreorderNlrTraversal traversal(AstProcessorManager::instance(), _stage);
    traversal.TraverseTranslationUnitDecl(tu);
}

SessionStage& TranspileASTConsumer::getSessionStage() {
    return _stage;
}

}  // namespace oklt
