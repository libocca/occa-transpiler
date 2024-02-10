#pragma once

#include <clang/AST/ASTConsumer.h>

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
