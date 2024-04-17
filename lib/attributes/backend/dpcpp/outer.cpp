#include "attributes/attribute_names.h"
#include "attributes/backend/dpcpp/common.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "core/attribute_manager/backend_handler.h"
#include "core/sema/okl_sema_ctx.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOuterAttribute(SessionStage& s,
                                  const clang::ForStmt& forStmt,
                                  const clang::Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@outer] attribute");

    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@outer params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@outer: failed to fetch loop meta data from sema"});
    }

    auto updatedParams = *params;
    // Auto Axis in loopInfo are replaced with specific. TODO: maybe somehow update params earlier?
    updatedParams.axis = loopInfo->axis.front();

    int openedScopeCounter = 0;
    auto prefixCode = dpcpp::buildInnerOuterLoopIdxLine(
        *loopInfo, updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    return replaceAttributedLoop(s, forStmt, a, suffixCode, prefixCode, true);
}

__attribute__((constructor)) void registerDpcppOuterAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        TargetBackend::DPCPP, OUTER_ATTR_NAME, handleOuterAttribute);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", OUTER_ATTR_NAME);
    }
}
}  // namespace
