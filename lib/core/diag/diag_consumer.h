#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/Basic/Diagnostic.h>
#include <llvm/Support/Registry.h>

namespace oklt {

class SessionStage;

class DiagConsumer : public clang::DiagnosticConsumer {
   public:
    explicit DiagConsumer(SessionStage& session);
    ~DiagConsumer() override = default;

    inline SessionStage& getSession() { return _session; };

    void HandleDiagnostic(clang::DiagnosticsEngine::Level Level,
                          const clang::Diagnostic& Info) override;

    bool IncludeInDiagnosticCounts() const override;

   protected:
    SessionStage& _session;
    std::atomic_flag _includeDiag = true;
};

}  // namespace oklt
