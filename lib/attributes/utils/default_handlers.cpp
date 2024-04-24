#include "attributes/utils/default_handlers.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <spdlog/spdlog.h>

namespace clang {
class Stmt;
}  // namespace clang

namespace oklt {
class SessionStage;

HandleResult emptyHandleStmtAttribute(SessionStage&, const clang::Stmt&, const clang::Attr& a) {
    SPDLOG_DEBUG("Called empty {} stmt handler", a.getNormalizedFullName());
    return {};
}

HandleResult emptyHandleDeclAttribute(SessionStage&, const clang::Decl&, const clang::Attr& a) {
    SPDLOG_DEBUG("Called empty {} decl handler", a.getNormalizedFullName());

    return {};
}

HandleResult defaultHandleSharedStmtAttribute(SessionStage& stage,
                                              const clang::Stmt& stmt,
                                              const clang::Attr& a) {
    SPDLOG_DEBUG("Called empty {} stmt handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* currLoop = sema.getLoopInfo();
    if (!currLoop) {
        return {};
    }

    currLoop->markSharedUsed();

    return {};
}

HandleResult defaultHandleExclusiveStmtAttribute(SessionStage& stage,
                                                 const clang::Stmt& stmt,
                                                 const clang::Attr& a) {
    SPDLOG_DEBUG("Called empty {} stmt handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* currLoop = sema.getLoopInfo();
    if (!currLoop) {
        return {};
    }

    currLoop->markExclusiveUsed();

    return {};
}

HandleResult defaultHandleSharedDeclAttribute(SessionStage& stage,
                                              const clang::Decl& d,
                                              const clang::Attr& a) {
    SPDLOG_DEBUG("Called empty {} decl handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* loopInfo = sema.getLoopInfo();
    if (loopInfo && loopInfo->isRegular()) {
        loopInfo = loopInfo->getAttributedParent();
    }
    if (!loopInfo) {
        return {};
    }

    loopInfo->sharedInfo.declared = true;

    return {};
}

HandleResult defaultHandleExclusiveDeclAttribute(SessionStage& stage,
                                                 const clang::Decl& d,
                                                 const clang::Attr& a) {
    SPDLOG_DEBUG("Called empty {} decl handler", a.getNormalizedFullName());

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto* loopInfo = sema.getLoopInfo();
    if (loopInfo && loopInfo->isRegular()) {
        loopInfo = loopInfo->getAttributedParent();
    }
    if (!loopInfo) {
        return {};
    }

    loopInfo->exclusiveInfo.declared = true;

    return {};
}

}  // namespace oklt
