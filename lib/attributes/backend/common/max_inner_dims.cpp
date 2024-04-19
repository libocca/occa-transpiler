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

HandleResult handleMaxInnerDimsStmtAttribute(SessionStage& s,
                                             const clang::ForStmt& forStmt,
                                             const clang::Attr& a,
                                             const AttributedLoopInnerSize* params) {
    SPDLOG_DEBUG("Handle [@max_inner_dims] attribute");
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@max_inner_dims params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (loopInfo && !loopInfo->parent) {
        bool isValidLoop = !loopInfo->isTiled() && loopInfo->is(LoopType::Outer);
        if (isValidLoop && !params->size.empty()) {
            OklLoopInfo::OptSizes sz = {1, 1, 1};
            for (size_t i = 0; i < params->size.size(); ++i) {
                auto s = params->size[i];
                if (s > 0) {
                    sz[i] = s;
                }
            }

            loopInfo->overridenInnerSizes = sz;
        }
    }

    removeAttribute(s, a);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = HandlerManager::instance().registerCommonHandler(MAX_INNER_DIMS, handleMaxInnerDimsStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("Failed to register {} attribute handler", MAX_INNER_DIMS);
    }
}
}  // namespace
