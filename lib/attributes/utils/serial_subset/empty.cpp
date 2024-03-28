#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

HandleResult handleEmptyStmtAttribute(const Attr& a,
                                      const Stmt& stmt,
                                      const std::any* params,
                                      SessionStage& s) {
    SPDLOG_DEBUG("Handle attribute [{}]", a.getNormalizedFullName());

    removeAttribute(a, s);
    return {};
}

HandleResult handleEmptyDeclAttribute(const Attr& a,
                                      const Decl& decl,
                                      const std::any* params,
                                      SessionStage& s) {
    SPDLOG_DEBUG("Handle attribute [{}]", a.getNormalizedFullName());

    removeAttribute(a, s);
    return {};
}

}  // namespace oklt::serial_subset
