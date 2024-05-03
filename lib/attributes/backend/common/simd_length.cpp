#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "core/handler_manager/attr_handler.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleSimdLengthStmtAttribute(SessionStage& s,
                                           const clang::ForStmt& forStmt,
                                           const clang::Attr& a,
                                           const AttributedLoopSimdLength* params) {
    SPDLOG_DEBUG("Handle [@simd_length] attribute");
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@simd_length params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (loopInfo && !loopInfo->parent) {
        if (loopInfo->has(LoopType::Outer) && params->size > 0) {
            loopInfo->simdLength = params->size;
        }
    }

    removeAttribute(s, a);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerCommonHandler(SIMD_LENGTH_NAME, handleSimdLengthStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("Failed to register {} attribute handler", SIMD_LENGTH_NAME);
    }
}
}  // namespace
