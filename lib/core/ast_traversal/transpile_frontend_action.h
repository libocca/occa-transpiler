#pragma once

#include "core/transpiler_session/transpiler_session.h"

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Frontend/FrontendAction.h>

namespace oklt {

class SessionStage;

class TranspileFrontendAction : public clang::ASTFrontendAction {
   public:
    explicit TranspileFrontendAction(TranspilerSession& session);
    ~TranspileFrontendAction() override = default;

   protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                          llvm::StringRef in_file) override;

    bool PrepareToExecuteAction(clang::CompilerInstance& compiler) override;

   private:
    TranspilerSession& _session;
    // INFO: it must leave longer than ASTConsumer for Diagnostic Consumer
    std::unique_ptr<SessionStage> _stage;
};
}  // namespace oklt
