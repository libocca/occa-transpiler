#include <clang/AST/Decl.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include "cuda_subset.h"

namespace oklt::cuda_like {
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

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::Function}, DeclHandler{handleGlobalFunction});

    if (!ok) {
        llvm::errs() << "failed to register implicit handler for global constant\n";
    }
}
}  // namespace oklt::cuda_like
