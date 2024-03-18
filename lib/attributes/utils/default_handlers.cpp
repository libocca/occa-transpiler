#include "attributes/utils/default_handlers.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

namespace clang {
class Stmt;
}  // namespace clang

namespace oklt {
class SessionStage;

HandleResult emptyHandleStmtAttribute(const clang::Attr& a,
                                             const clang::Stmt&,
                                             SessionStage&) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "Called empty " << a.getNormalizedFullName() << " stmt handler\n";
#endif
    return {};
}

HandleResult emptyHandleDeclAttribute(const clang::Attr& a,
                                             const clang::Decl&,
                                             SessionStage&) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "Called empty " << a.getNormalizedFullName() << " decl handler\n";
#endif
    return {};
}

HandleResult defaultHandleSharedStmtAttribute(const clang::Attr& a,
                                              const clang::Stmt& stmt,
                                              SessionStage& stage) {
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* currLoop = sema.getLoopInfo();
    if (!currLoop) {
        return {};
    }
    currLoop->markShmUsed();
    return {};
}

}  // namespace oklt
