#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>
namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string restrictText = "__restrict__ ";
}  // namespace

HandleResult handleRestrictAttribute(const clang::Attr& a,
                                     const clang::Decl& decl,
                                     SessionStage& s) {
    SPDLOG_DEBUG("Handle [@restrict] attribute");

    removeAttribute(a, s);
    s.getRewriter().InsertTextBefore(decl.getLocation(), restrictText);
    return {};
}

}  // namespace oklt::serial_subset
