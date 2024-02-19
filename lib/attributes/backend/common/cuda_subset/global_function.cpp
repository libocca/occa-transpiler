#include "core/attribute_manager/attribute_manager.h"
#include "attributes/backend/common/cuda_subset/cuda_subset.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>


namespace oklt::cuda_subset {
using namespace oklt;
using namespace clang;

bool handleGlobalFunction(const clang::Decl* decl, SessionStage& s) {
    //   Check if function
    if (!isa<FunctionDecl>(decl)) {
        return true;
    }

    //   Check if function is not attributed with OKL attribute
    auto& am = s.getAttrManager();
    if ((decl->hasAttrs()) && (am.checkAttrs(decl->getAttrs(), decl, s))) {
        return true;
    }

    auto& rewriter = s.getRewriter();
    auto loc = decl->getSourceRange().getBegin();
    rewriter.InsertTextBefore(loc, "__device__ ");  // Replace 'const' with __constant__
    return true;
}

}  // namespace oklt::cuda_subset
