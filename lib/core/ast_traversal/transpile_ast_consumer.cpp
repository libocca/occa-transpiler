#include "oklt/core/ast_traversal/transpile_ast_consumer.h"

namespace oklt {
using namespace clang;

TranspileASTConsumer::TranspileASTConsumer(
    TranspilerConfig &&config,
    ASTContext &ctx)
    :_config(std::move(config))
    ,_session(_config, ctx)
    ,_visitor(_session)
{}

void TranspileASTConsumer::HandleTranslationUnit(ASTContext &context)
{
  TranslationUnitDecl *tu = context.getTranslationUnitDecl();
  _visitor.TraverseDecl(tu);
  _session.writeTranspiledSource();
}

}
