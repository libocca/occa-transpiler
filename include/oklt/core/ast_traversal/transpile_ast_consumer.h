#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
// #include "oklt/core/ast_traversal/ast_visitor.h"
#include <oklt/core/ast_traversal/semantic_analyzer.h>

namespace oklt {

class SessionStage;

class TranspileASTConsumer : public clang::ASTConsumer {
 public:
  explicit TranspileASTConsumer(SessionStage& stage);
  void HandleTranslationUnit(clang::ASTContext& context) override;

  SessionStage& getSessionStage();
  SemanticAnalyzer& getSemaAnalyzer();
 private:
  SessionStage& _stage;
  SemanticAnalyzer _semaAnalyzer;
};

}  // namespace oklt
