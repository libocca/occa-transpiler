#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

namespace {
const std::string SHARED_MODIFIER = "__shared__";
}
namespace oklt::cuda_subset {
tl::expected<std::any, Error> handleSharedAttribute(const clang::Attr* a,
                                                    const clang::Decl* d,
                                                    const std::any& params,
                                                    SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();
    removeAttribute(a, s);
    std::string replacedAttribute = " " + SHARED_MODIFIER + " ";
    return rewriter.InsertText(d->getBeginLoc(), replacedAttribute, false, false);
}
}  // namespace oklt::cuda_subset
