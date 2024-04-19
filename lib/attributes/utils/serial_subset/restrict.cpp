#include "core/handler_manager/handler_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string restrictText = "__restrict__ ";
}  // namespace

HandleResult handleRestrictAttribute(SessionStage& s,
                                     const clang::Decl& decl,
                                     const clang::Attr& a) {
    SPDLOG_DEBUG("Handle [@restrict] attribute");

    removeAttribute(s, a);
    s.getRewriter().InsertTextBefore(decl.getLocation(), restrictText);
    return {};
}

}  // namespace oklt::serial_subset
