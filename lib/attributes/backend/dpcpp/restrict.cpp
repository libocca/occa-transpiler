#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;
const std::string RESTRICT_MODIFIER = "__restrict__ ";

HandleResult handleRestrictAttribute(SessionStage& s,
                                     const clang::Decl& decl,
                                     const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@restrict] attribute");
    removeAttribute(a, s);
    s.getRewriter().InsertTextBefore(decl.getLocation(), RESTRICT_MODIFIER);
    return {};
}

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, RESTRICT_ATTR_NAME, ASTNodeKind::getFromNodeKind<Decl>()},
        makeSpecificAttrHandle(handleRestrictAttribute));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", RESTRICT_ATTR_NAME);
    }
}
}  // namespace
