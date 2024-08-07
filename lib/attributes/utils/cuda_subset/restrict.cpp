#include "core/handler_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <spdlog/spdlog.h>

namespace {
const std::string RESTRICT_MODIFIER = "__restrict__ ";
}
namespace oklt::cuda_subset {
using namespace clang;

HandleResult handleRestrictAttribute(SessionStage& s, const Decl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@restrict] attribute");

    removeAttribute(s, a);
    auto kind = decl.getKind();
    if (isa<VarDecl, FieldDecl, FunctionDecl>(decl)) {
        s.getRewriter().InsertTextBefore(decl.getLocation(), RESTRICT_MODIFIER);
    }

    return {};
}

}  // namespace oklt::cuda_subset
