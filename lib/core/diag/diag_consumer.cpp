#include "core/diag/diag_consumer.h"
#include "core/diag/diag_handler.h"
#include "core/transpiler_session/session_stage.h"

#include <llvm/Support/ManagedStatic.h>

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
    : _session(session),
      clang::DiagnosticConsumer(){};

void DiagConsumer::HandleDiagnostic(DiagnosticsEngine::Level DiagLevel, const Diagnostic& Info) {
    if (!_includeDiag.test_and_set()) {
        return;
    }

    auto msg = StoredDiagnostic(DiagLevel, Info);
    _session.pushDiagnosticMessage(msg);

    DiagnosticConsumer::HandleDiagnostic(DiagLevel, Info);
}

bool DiagConsumer::IncludeInDiagnosticCounts() const {
    auto& diag = _session.getCompiler().getDiagnostics();

    Diagnostic info(&diag);
    auto diagLevel = diag.getDiagnosticLevel(info.getID(), info.getLocation());

    // Accept only Warning, Error and Fatal
    if (diagLevel < DiagnosticsEngine::Level::Warning) {
        const_cast<std::atomic_flag&>(_includeDiag).clear();
        return false;
    }

    for (auto& ptr : getDiagDiagHandleInstances()) {
        if (ptr->_id == info.getID() && ptr->HandleDiagnostic(_session, diagLevel, info)) {
            const_cast<std::atomic_flag&>(_includeDiag).clear();
            return false;
        }
    }

    return true;
}

}  // namespace oklt
