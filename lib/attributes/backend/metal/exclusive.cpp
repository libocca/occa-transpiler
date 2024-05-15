#include "attributes/backend/metal/common.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleExclusiveDeclAttribute(SessionStage& s, const Decl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute (Decl)");

    removeAttribute(s, a);
    return {};
}

HandleResult handleExclusiveVarAttribute(SessionStage& s, const VarDecl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute");

    removeAttribute(s, a);

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (loopInfo && loopInfo->isRegular()) {
        loopInfo = loopInfo->getAttributedParent();
    }
    if (loopInfo && loopInfo->has(LoopType::Inner)) {
        return tl::make_unexpected(
            Error{{}, "Cannot define [@exclusive] variables inside an [@inner] loop"});
    }

    auto child = loopInfo ? loopInfo->getFirstAttributedChild() : nullptr;
    bool isInnerChild = child && child->has(LoopType::Inner);
    if (!loopInfo || !loopInfo->has(LoopType::Outer) || !isInnerChild) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    return defaultHandleExclusiveDeclAttribute(s, decl, a);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerBackendHandler(
        TargetBackend::METAL, EXCLUSIVE_ATTR_NAME, handleExclusiveDeclAttribute);
    ok &= registerBackendHandler(
        TargetBackend::METAL, EXCLUSIVE_ATTR_NAME, handleExclusiveVarAttribute);
    ok &= registerBackendHandler(
        TargetBackend::METAL, EXCLUSIVE_ATTR_NAME, defaultHandleExclusiveStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", EXCLUSIVE_ATTR_NAME);
    }
}
}  // namespace
