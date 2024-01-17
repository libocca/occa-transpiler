#pragma once

#include <clang/AST/ASTConsumer.h>
#include "oklt/core/transpile_session/transpile_session.h"
#include "oklt/core/ast_traversal/ast_visitor.h"
#include "oklt/core/config.h"

namespace oklt {

class TranspileASTConsumer : public clang::ASTConsumer {
public:
  TranspileASTConsumer(TranspilerConfig &&config,
                       clang::ASTContext &ctx);
  void HandleTranslationUnit(clang::ASTContext &context) override;
private:
  TranspilerConfig _config;
  TranspileSession _session;
  ASTVisitor _visitor;
};

}
