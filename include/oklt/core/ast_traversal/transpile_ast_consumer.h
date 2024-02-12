#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
#include "oklt/core/ast_traversal/ast_visitor.h"

namespace oklt {

class SessionStage;

class TranspileASTConsumer : public clang::ASTConsumer {
 public:
  explicit TranspileASTConsumer(SessionStage& stage);
  void HandleTranslationUnit(clang::ASTContext& context) override;

  SessionStage& getSessionStage();

 private:
  SessionStage& _stage;
};

}  // namespace oklt
