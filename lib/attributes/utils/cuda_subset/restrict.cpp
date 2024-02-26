#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <clang/AST/Attr.h>

namespace {
const std::string RESTRICT_MODIFIER = "__restrict__";
}
namespace oklt::cuda_subset {
using namespace clang;
HandleResult handleRestrictAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();

    if (!isa<ParmVarDecl>(d)) {
        return false;
    }

    auto parmDecl = cast<ParmVarDecl>(d);
    SourceLocation identifierLoc = parmDecl->getLocation();
    removeAttribute(a, s);
    std::string restrictText = " " + RESTRICT_MODIFIER + " ";
    rewriter.InsertText(identifierLoc, restrictText, false, false);

    // INFO: might be better to use rewriter.getRewrittenText() method

    auto& ctx = parmDecl->getASTContext();
    SourceRange r1(parmDecl->getSourceRange().getBegin(), identifierLoc);
    auto part1 = getSourceText(r1, ctx);
    auto ident = parmDecl->getQualifiedNameAsString();
    std::string modifiedArgument = part1 + restrictText + ident;

    if (s.getAstProccesorType() == AstProcessorType::OKL_WITH_SEMA) {
        s.tryEmplaceUserCtx<OklSemaCtx>().setKernelArgRawString(parmDecl, modifiedArgument);
    }

    return true;
}

}  // namespace oklt::cuda_subset
