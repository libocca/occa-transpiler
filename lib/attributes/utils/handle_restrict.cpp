#include "attributes/utils/handle_restrict.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <clang/AST/Attr.h>

namespace oklt {
using namespace clang;
bool handleRestrictAttribute(const clang::Attr* a,
                             const clang::Decl* d,
                             SessionStage& s,
                             const std::string& replaceQualifier) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();

    if (!isa<VarDecl>(d)) {
        return false;
    }

    auto varDecl = cast<VarDecl>(d);
    SourceLocation identifierLoc = varDecl->getLocation();
    removeAttribute(a, s);
    std::string restrictText = " " + replaceQualifier + " ";
    rewriter.InsertText(identifierLoc, restrictText, false, false);

    // INFO: might be better to use rewriter.getRewrittenText() method

    auto semaCtx = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto parsingInfo = semaCtx.getParsingKernelInfo();
    if (!parsingInfo) {
        // TODO: internal error
        return false;
    }
    auto& ctx = varDecl->getASTContext();
    SourceRange r1(varDecl->getSourceRange().getBegin(), identifierLoc);
    auto part1 = getSourceText(r1, ctx);
    auto ident = varDecl->getQualifiedNameAsString();
    std::string modifiedArgument = part1 + restrictText + ident;
    parsingInfo->argStrs.push_back(modifiedArgument);
    return true;
}

}  // namespace oklt
