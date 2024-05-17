#include "attributes/backend/metal/common.h"
#include "attributes/frontend/params/loop.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOuterAttribute(SessionStage& s,
                                  const clang::ForStmt& forStmt,
                                  const clang::Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@outer] attribute");

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{
            .ec = std::error_code(), .desc = "@outer: failed to fetch loop meta data from sema"});
    }

    // Auto Axis in loopInfo are replaced with specific.
    // TODO: maybe somehow update params earlier?
    auto updatedParams = *params;
    updatedParams.axis = loopInfo->axis.front();

    int openedScopeCounter = 0;
    auto prefixCode = metal::buildInnerOuterLoopIdxLine(
        *loopInfo, updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    return replaceAttributedLoop(s, forStmt, a, suffixCode, prefixCode, true);
}

__attribute__((constructor)) void registerBackendHandler() {
    auto ok = registerBackendHandler(TargetBackend::METAL, OUTER_ATTR_NAME, handleOuterAttribute);

    if (!ok) {
        SPDLOG_ERROR("[METAL] Failed to register {} attribute handler", OUTER_ATTR_NAME);
    }
}
}  // namespace
