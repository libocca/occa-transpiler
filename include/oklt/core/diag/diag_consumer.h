#pragma once

#include <clang/Basic/Diagnostic.h>
#include <clang/AST/ASTContext.h>
#include <llvm/Support/Registry.h>

namespace oklt {

class SessionStage;

class DiagConsumer : public clang::DiagnosticConsumer {
public:
  explicit DiagConsumer(SessionStage &session);
  ~DiagConsumer() override = default;

  inline SessionStage &getSession() { return _session; };

  void HandleDiagnostic(clang::DiagnosticsEngine::Level Level, const clang::Diagnostic &Info) override;

protected:
  SessionStage &_session;
};

}
