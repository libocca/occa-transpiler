#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

namespace oklt::cuda_subset {
HandleResult handleExclusiveAttribute(const clang::Attr& a, const clang::Decl& d, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();
    removeAttribute(a, s);
    return true;
}
}  // namespace oklt::cuda_subset
