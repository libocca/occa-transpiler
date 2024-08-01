#include "core/handler_manager/handler_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string RESTRICT_MODIFIER = "__restrict__ ";
}  // namespace

HandleResult handleRestrictAttribute(SessionStage& s, const Decl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@restrict] attribute");

    removeAttribute(s, a);
    if (isa<VarDecl, FieldDecl, FunctionDecl>(decl)) {
        s.getRewriter().InsertTextBefore(decl.getLocation(), RESTRICT_MODIFIER);
    }

    return {};
}

}  // namespace oklt::serial_subset
