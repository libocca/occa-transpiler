#pragma once

#include <clang/Basic/Diagnostic.h>
#include <clang/AST/ASTContext.h>
#include <llvm/Support/Registry.h>

namespace oklt {

class SessionStage;

class DiagConsumer : public clang::DiagnosticConsumer {
public:
  DiagConsumer(SessionStage &session);
  ~DiagConsumer() override = default;

  void HandleDiagnostic(clang::DiagnosticsEngine::Level Level, const clang::Diagnostic &Info) override;

protected:
  SessionStage &_session;
};

}
