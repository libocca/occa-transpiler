#include "core/diag/diag_handler.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <llvm/Support/ManagedStatic.h>

namespace {
using namespace oklt;
using namespace clang;

class IgnoreUndeclHandler : public DiagHandler {
   public:
    IgnoreUndeclHandler()
        : DiagHandler(diag::err_undeclared_var_use){};

    bool HandleDiagnostic(SessionStage& session, DiagLevel level, const Diagnostic& info) override {
        // TODO unify with error reporting to get correct location and source line
        llvm::SmallString<64> buf;
        info.FormatDiagnostic(buf);
        std::string msg{buf.begin(), buf.end()};

        FullSourceLoc loc(info.getLocation(), session.getCompiler().getSourceManager());

        session.pushWarning(std::to_string(loc.getLineNumber()) + ":" + msg.data());
        return true;
    }
};

oklt::DiagHandlerRegistry::Add<IgnoreUndeclHandler> diag_dim("IgnoreUndeclUse", "");
}  // namespace
