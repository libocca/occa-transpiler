#include "core/attribute_manager/result.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/DeclBase.h>

namespace {
const std::string SHARED_MODIFIER = "__shared__";
}
namespace oklt::cuda_subset {
HandleResult handleSharedAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
    // auto& rewriter = s.getRewriter();
    // removeAttribute(a, s);
    std::string replacedAttribute = " " + SHARED_MODIFIER + " ";

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a->getNormalizedFullName(), 1)
        .addReplacement(OKL_TRANSPILED_ATTR, getAttrFullSourceRange(*a), replacedAttribute)
        .build();

    // return rewriter.InsertText(d->getBeginLoc(), replacedAttribute, false, false);
}
}  // namespace oklt::cuda_subset
