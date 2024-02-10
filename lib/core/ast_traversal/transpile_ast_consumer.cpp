#include <oklt/core/ast_traversal/transpile_ast_consumer.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/ast_traversal/ast_traversal.hpp>

#ifdef USE_MOCK_SEMA
#include <oklt/core/ast_traversal/sema/mock_processor.hpp>
#else
#include <oklt/core/ast_traversal/sema/sema.hpp>
#endif

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _stage(stage)
{}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
#ifdef USE_MOCK_SEMA
  AstTraversal<SemanticMockProcessor> astTraversal(_stage);
#else
  AstTraversal<SemanticProcessor> astTraversal(_stage);
#endif
  TranslationUnitDecl* tu = context.getTranslationUnitDecl();
  astTraversal.TraverseDecl(tu);
}

SessionStage& TranspileASTConsumer::getSessionStage() {
  return _stage;
}

}  // namespace oklt
