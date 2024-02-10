#include <oklt/core/ast_traversal/transpile_ast_consumer.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/ast_traversal/ast_traversal.hpp>
#include <oklt/core/ast_traversal/sema/mock_processor.hpp>

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _stage(stage)
{}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
  AstTraversal<SemanticMockProcessor> astTraversal(_stage);
  TranslationUnitDecl* tu = context.getTranslationUnitDecl();
  astTraversal.TraverseDecl(tu);
}

SessionStage& TranspileASTConsumer::getSessionStage() {
  return _stage;
}

}  // namespace oklt
