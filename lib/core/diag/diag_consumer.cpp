#include "oklt/core/diag/diag_consumer.h"
#include "oklt/core/diag/diag_handler.h"
#include "oklt/core/transpiler_session/transpiler_session.h"

#include "llvm/Support/ManagedStatic.h"

LLVM_INSTANTIATE_REGISTRY(oklt::DiagHandlerRegistry);

namespace oklt {
using namespace clang;

static const std::list<std::unique_ptr<DiagHandler>>& getDiagDiagHandleInstances() {
  static llvm::ManagedStatic<std::list<std::unique_ptr<DiagHandler>>> DiagHandleInstances;
  if (!DiagHandleInstances->empty())
    return *DiagHandleInstances;

  for (const auto& It : DiagHandlerRegistry::entries()) {
    DiagHandleInstances->emplace_back(It.instantiate());
  }

  return *DiagHandleInstances;
}

DiagConsumer::DiagConsumer(SessionStage& session)
    : _session(session), clang::DiagnosticConsumer(){};

void DiagConsumer::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel, const Diagnostic& Info) {
  // Accept only Warning, Error and Fatal
  if (DiagLevel < DiagnosticsEngine::Level::Warning)
    return;

  for (auto& Ptr : getDiagDiagHandleInstances()) {
    if (Ptr->_id == Info.getID() && Ptr->HandleDiagnostic(_session, DiagLevel, Info))
      return;
  }

  auto msg = StoredDiagnostic(DiagLevel, Info);
  _session.pushDiagnosticMessage(msg);

  DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);
}

}  // namespace oklt
