#include "attributes/attribute_names.h"
#include "attributes/utils/default_handlers.h"
#include "core/handler_manager/backend_handler.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;
const std::string RESTRICT_MODIFIER = "__restrict__ ";

HandleResult handleRestrictAttribute(SessionStage& s, const Decl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@restrict] attribute");

    removeAttribute(s, a);
    if (isa<VarDecl, FieldDecl, FunctionDecl>(decl)) {
        s.getRewriter().InsertTextBefore(decl.getLocation(), RESTRICT_MODIFIER);
    }

    return {};
}

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok =
        registerBackendHandler(TargetBackend::DPCPP, RESTRICT_ATTR_NAME, handleRestrictAttribute);

    ok &= registerBackendHandler(TargetBackend::DPCPP, RESTRICT_ATTR_NAME, emptyHandleStmtAttribute);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", RESTRICT_ATTR_NAME);
    }
}
}  // namespace
