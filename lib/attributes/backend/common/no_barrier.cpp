#include "attributes/attribute_names.h"
#include "core/handler_manager/attr_handler.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleNoBarrierStmtAttribute(SessionStage& s,
                                          const clang::ForStmt& forStmt,
                                          const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@nobarrier] attribute");
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@nobarrier: failed to fetch loop meta data from sema"});
    }

    loopInfo->sharedInfo.used = false;

    removeAttribute(s, a);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok =
        HandlerManager::registerCommonHandler(NO_BARRIER_ATTR_NAME, handleNoBarrierStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("Failed to register {} attribute handler", NO_BARRIER_ATTR_NAME);
    }
}
}  // namespace
