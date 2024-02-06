#include "oklt/core/ast_traversal/transpile_ast_consumer.h"
#include "oklt/core/transpiler_session/session_stage.h"
#include "oklt/core/ast_traversal/semantic_category.h"

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& stage)
    : _stage(stage),
      _semaAnalyzer(fromBackendType(_stage.getBackend()), stage)
{}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
  TranslationUnitDecl* tu = context.getTranslationUnitDecl();
  _semaAnalyzer.TraverseDecl(tu);
}

SessionStage& TranspileASTConsumer::getSessionStage() {
  return _stage;
}

SemanticAnalyzer& TranspileASTConsumer::getSemaAnalyzer() {
  return _semaAnalyzer;
}

}  // namespace oklt
