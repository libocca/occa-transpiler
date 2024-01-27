#include "oklt/core/ast_traversal/transpile_ast_consumer.h"

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(SessionStage& session)
    : _session(session), _visitor(_session) {}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext& context) {
  TranslationUnitDecl* tu = context.getTranslationUnitDecl();
  _visitor.TraverseDecl(tu);
}

}  // namespace oklt
