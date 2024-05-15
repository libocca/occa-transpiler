#include "attributes/backend/metal/common.h"
#include "attributes/frontend/params/loop.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleInnerAttribute(SessionStage& s,
                                  const clang::ForStmt& forStmt,
                                  const clang::Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@inner] attribute");
    handleChildAttr(s, forStmt, NO_BARRIER_ATTR_NAME);

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{std::error_code(), "@inner: failed to fetch loop meta data from sema"});
    }

    // Auto Axis in loopInfo are replaced with specific.
    // TODO: maybe somehow update params earlier?
    auto updatedParams = *params;
    updatedParams.axis = loopInfo->axis.front();

    std::string afterRBraceCode = "";
    if (loopInfo->shouldSync()) {
        afterRBraceCode += metal::SYNC_THREADS_BARRIER + ";\n";
    }

    int openedScopeCounter = 0;
    auto prefixCode = metal::buildInnerOuterLoopIdxLine(
        *loopInfo, updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    return replaceAttributedLoop(s, forStmt, a, suffixCode, afterRBraceCode, prefixCode, true);
}

__attribute__((constructor)) void registerBackendHandler() {
    auto ok = registerBackendHandler(TargetBackend::METAL, INNER_ATTR_NAME, handleInnerAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", INNER_ATTR_NAME);
    }
}
}  // namespace
