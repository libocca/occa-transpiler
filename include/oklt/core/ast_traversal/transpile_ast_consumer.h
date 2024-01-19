#pragma once

#include <clang/AST/ASTConsumer.h>
#include "oklt/core/transpiler_session/transpiler_session.h"
#include "oklt/core/ast_traversal/ast_visitor.h"

namespace oklt {

class TranspileASTConsumer : public clang::ASTConsumer {
public:
  explicit TranspileASTConsumer(SessionStage session);
  void HandleTranslationUnit(clang::ASTContext &context) override;
private:
  SessionStage _session;
  ASTVisitor _visitor;
};

}
