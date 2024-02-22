#include "attributes/utils/handle_shared.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

namespace oklt {
bool handleSharedAttribute(const clang::Attr* a,
                           const clang::Decl* d,
                           SessionStage& s,
                           const std::string& replaceQualifier) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();
    removeAttribute(a, s);
    std::string replacedAttribute = " " + replaceQualifier + " ";
    return rewriter.InsertText(d->getBeginLoc(), replacedAttribute, false, false);
}
}  // namespace oklt
