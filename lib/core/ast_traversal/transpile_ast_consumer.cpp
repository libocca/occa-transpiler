#include "oklt/core/ast_traversal/transpile_ast_consumer.h"
#include "oklt/core/transpiler_session/session_stage.h"

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _session(stage), _visitor(_session) {}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
  TranslationUnitDecl* tu = context.getTranslationUnitDecl();
  _visitor.TraverseDecl(tu);
}

SessionStage& TranspileASTConsumer::getSessionStage() {
  return _session;
}

ASTVisitor& TranspileASTConsumer::getAstVisitor() {
  return _visitor;
}

}  // namespace oklt
