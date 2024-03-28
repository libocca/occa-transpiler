#include "attributes/utils/default_handlers.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <spdlog/spdlog.h>

namespace clang {
class Stmt;
}  // namespace clang

namespace oklt {
class SessionStage;

HandleResult emptyHandleStmtAttribute(const clang::Attr& a, const clang::Stmt&, SessionStage&) {
    SPDLOG_DEBUG("Called empty {} stmt handler", a.getNormalizedFullName());
    return {};
}

HandleResult emptyHandleDeclAttribute(const clang::Attr& a, const clang::Decl&, SessionStage&) {
    SPDLOG_DEBUG("Called empty {} decl handler", a.getNormalizedFullName());

    return {};
}

HandleResult defaultHandleSharedStmtAttribute(const clang::Attr& a,
                                              const clang::Stmt& stmt,
                                              SessionStage& stage) {
    SPDLOG_DEBUG("Called empty {} stmt handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* currLoop = sema.getLoopInfo();
    if (!currLoop) {
        return {};
    }
    currLoop->markSharedUsed();
    return {};
}

HandleResult defaultHandleExclusiveStmtAttribute(const clang::Attr& a,
                                                 const clang::Stmt& stmt,
                                                 SessionStage& stage) {
    SPDLOG_DEBUG("Called empty {} stmt handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* currLoop = sema.getLoopInfo();
    if (!currLoop) {
        return {};
    }
    currLoop->markExclusiveUsed();
    return {};
}

HandleResult defaultHandleSharedDeclAttribute(const clang::Attr& a,
                                              const clang::Decl& d,
                                              SessionStage& stage) {
    SPDLOG_DEBUG("Called empty {} decl handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* currLoop = sema.getLoopInfo();
    if (!currLoop) {
        return {};
    }
    currLoop->sharedInfo.declared = true;

    return {};
}

HandleResult defaultHandleExclusiveDeclAttribute(const clang::Attr& a,
                                                 const clang::Decl& d,
                                                 SessionStage& stage) {
    SPDLOG_DEBUG("Called empty {} decl handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* currLoop = sema.getLoopInfo();
    if (!currLoop) {
        return {};
    }
    currLoop->exclusiveInfo.declared = true;
    return {};
}

}  // namespace oklt
