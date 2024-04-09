#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

HandleResult handleEmptyStmtAttribute(SessionStage& s,
                                      const Stmt& stmt,
                                      const Attr& a,
                                      const std::any* params) {
    SPDLOG_DEBUG("Handle attribute [{}]", a.getNormalizedFullName());

    removeAttribute(s, a);
    return {};
}

HandleResult handleEmptyDeclAttribute(SessionStage& s,
                                      const Decl& decl,
                                      const Attr& a,
                                      const std::any* params) {
    SPDLOG_DEBUG("Handle attribute [{}]", a.getNormalizedFullName());

    removeAttribute(s, a);
    return {};
}

}  // namespace oklt::serial_subset
