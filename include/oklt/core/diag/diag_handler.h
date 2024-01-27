#pragma once

#include <clang/AST/ASTContext.h>
#include <clang/Basic/Diagnostic.h>
#include <llvm/Support/Registry.h>

namespace oklt {

class SessionStage;
class DiagConsumer;
using DiagLevel = clang::DiagnosticsEngine::Level;

class DiagHandler {
  friend class DiagConsumer;

 public:
  explicit DiagHandler(unsigned id) : _id(id){};
  virtual ~DiagHandler() = default;
  virtual bool HandleDiagnostic(SessionStage& session,
                                DiagLevel level,
                                const clang::Diagnostic& info) = 0;

 protected:
  unsigned _id = 0;
};

typedef llvm::Registry<DiagHandler> DiagHandlerRegistry;
}  // namespace oklt
