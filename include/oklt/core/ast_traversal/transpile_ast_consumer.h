#pragma once

#include <clang/AST/ASTConsumer.h>
#include <clang/Frontend/CompilerInstance.h>
#include <oklt/core/ast_traversal/semantic_base.h>
#include <memory>

namespace oklt {

class SessionStage;

class TranspileASTConsumer : public clang::ASTConsumer {
 public:
  explicit TranspileASTConsumer(SessionStage& stage);
  void HandleTranslationUnit(clang::ASTContext& context) override;

  SessionStage& getSessionStage();

  template<class T>
  T * getSemaAnalyzer() {
    T * target = dynamic_cast<T*>(_semaAnalyzer.get());
    assert(target && "Used unexpected Semantic Analyzer type");
    return target;
  }

 private:
  SessionStage& _stage;
  std::unique_ptr<SemanticASTVisitorBase> _semaAnalyzer;
};

}  // namespace oklt
