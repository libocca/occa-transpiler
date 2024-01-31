#pragma once

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
// #include "oklt/core/transpiler_session/transpiler_session.h"

namespace oklt {
class TranspilerSession;
class SessionStage;
}

namespace oklt {
class TranspileFrontendAction : public clang::ASTFrontendAction {
 public:
  explicit TranspileFrontendAction(TranspilerSession& session);
  ~TranspileFrontendAction() override = default;

 protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                        llvm::StringRef in_file) override;
private:
  TranspilerSession& _session;
  //INFO: it must leave longer than ASTConsumer for Diagnostic Consumer
  std::unique_ptr<SessionStage> _stage;
};
}  // namespace oklt
