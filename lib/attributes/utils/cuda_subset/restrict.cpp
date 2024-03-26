#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>

namespace {
const std::string RESTRICT_MODIFIER = "__restrict__ ";
}
namespace oklt::cuda_subset {
using namespace clang;
HandleResult handleRestrictAttribute(const clang::Attr& a,
                                     const clang::Decl& decl,
                                     SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    removeAttribute(a, s);
    s.getRewriter().InsertTextBefore(decl.getLocation(), RESTRICT_MODIFIER);

    return {};
}

}  // namespace oklt::cuda_subset
