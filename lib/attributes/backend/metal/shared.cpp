#include "attributes/backend/metal/common.h"

#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string SHARED_MODIFIER = "threadgroup";

HandleResult handleSharedDeclAttribute(SessionStage& s, const Decl& var, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    return removeAttribute(s, a);
}

HandleResult handleSharedTypeAttribute(SessionStage& s, const TypedefDecl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    removeAttribute(s, a);

    auto loc = decl.getTypeSourceInfo()->getTypeLoc().getBeginLoc();
    s.getRewriter().InsertTextBefore(loc, SHARED_MODIFIER + " ");

    return {};
}

HandleResult handleSharedVarAttribute(SessionStage& s, const VarDecl& d, const Attr& a) {
    SPDLOG_DEBUG("Handle [@shared] attribute");

    removeAttribute(s, a);

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (loopInfo && loopInfo->isRegular()) {
        loopInfo = loopInfo->getAttributedParent();
    }
    if (loopInfo && loopInfo->has(LoopType::Inner)) {
        return tl::make_unexpected(
            Error{{}, "Cannot define [@shared] variables inside an [@inner] loop"});
    }
    auto child = loopInfo ? loopInfo->getFirstAttributedChild() : nullptr;
    bool isInnerChild = child && child->has(LoopType::Inner);

    // This diagnostic is applied only to variable declaration
    if (!loopInfo || !loopInfo->has(LoopType::Outer) || !isInnerChild) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    s.getRewriter().InsertTextBefore(d.getTypeSpecStartLoc(), SHARED_MODIFIER + " ");

    return defaultHandleSharedDeclAttribute(s, d, a);
}

__attribute__((constructor)) void registerCUDASharedAttrBackend() {
    auto ok =
        registerBackendHandler(TargetBackend::METAL, SHARED_ATTR_NAME, handleSharedDeclAttribute);
    ok &= registerBackendHandler(TargetBackend::METAL, SHARED_ATTR_NAME, handleSharedTypeAttribute);
    ok &= registerBackendHandler(TargetBackend::METAL, SHARED_ATTR_NAME, handleSharedVarAttribute);

    // Empty Stmt handler since @shared variable is of attributed type, it is called on DeclRefExpr
    ok &= registerBackendHandler(
        TargetBackend::METAL, SHARED_ATTR_NAME, defaultHandleSharedStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", SHARED_ATTR_NAME);
    }
}
}  // namespace
