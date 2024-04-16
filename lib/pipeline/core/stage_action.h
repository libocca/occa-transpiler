#pragma once

#include "core/rewriter/rewriter_fabric.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/Frontend/FrontendAction.h>

namespace clang {
class CompilerInstance;
}

namespace oklt {
/**
 * @brief Base stage action file tp run a transpiler pipeline
 */
class StageAction : public clang::ASTFrontendAction {
   public:
    StageAction() = default;

    StageAction(const StageAction&) = delete;
    StageAction& operator=(const StageAction&) = delete;

    bool setSession(SharedTranspilerSession session);

    bool PrepareToExecuteAction(clang::CompilerInstance& compiler) override;
    void EndSourceFileAction() override;

   protected:
    virtual RewriterProxyType getRewriterType() const { return RewriterProxyType::Original; }

    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& CI,
                                                          llvm::StringRef InFile) override {
        StageAction::EndSourceFileAction();
        return nullptr;
    }

    std::unique_ptr<SessionStage> _stage;
    SharedTranspilerSession _session;
    std::string _name;
};

}  // namespace oklt
