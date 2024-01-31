#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
#include "oklt/core/ast_traversal/ast_visitor.h"

namespace oklt {
class SessionStage;
}

namespace oklt {

class TranspileASTConsumer : public clang::ASTConsumer {
 public:
  explicit TranspileASTConsumer(SessionStage& session);
  void HandleTranslationUnit(clang::ASTContext& context) override;

  SessionStage &getSessionStage();
  ASTVisitor &getAstVisitor();
 private:
  SessionStage &_session;
  ASTVisitor _visitor;

};

}  // namespace oklt
