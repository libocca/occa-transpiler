#pragma once

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>
#include "oklt/core/transpiler_session/transpiler_session.h"

namespace oklt {
class TranspileFrontendAction : public clang::ASTFrontendAction {
 public:
  explicit TranspileFrontendAction(TranspilerSession& session);
  ~TranspileFrontendAction() override = default;

 protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                        llvm::StringRef in_file) override;
  void ExecuteAction() override;

 private:
  TranspilerSession& _session;
  std::unique_ptr<SessionStage> _stage = nullptr;
  std::unique_ptr<clang::DiagnosticConsumer> _diag = nullptr;
};
}  // namespace oklt
