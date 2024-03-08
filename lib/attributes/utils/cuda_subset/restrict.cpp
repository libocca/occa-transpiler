#include "core/attribute_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <clang/AST/Attr.h>

namespace {
const std::string RESTRICT_MODIFIER = "__restrict__";
}
namespace oklt::cuda_subset {
using namespace clang;
HandleResult handleRestrictAttribute(const clang::Attr& a,
                                     const clang::Decl& decl,
                                     SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    SourceLocation identifierLoc = decl.getLocation();
    std::string restrictText = " " + RESTRICT_MODIFIER + " ";
    // INFO: might be better to use rewriter.getRewrittenText() method

    auto& ctx = decl.getASTContext();
    SourceRange r1(decl.getSourceRange().getBegin(), identifierLoc);
    auto part1 = getSourceText(r1, ctx);

    auto ident = [](const clang::Decl& decl) {
        switch (decl.getKind()) {
            case Decl::Field:
                return cast<FieldDecl>(decl).getQualifiedNameAsString();
            default:
                return cast<VarDecl>(decl).getQualifiedNameAsString();
        }
    }(decl);

    std::string modifiedArgument = part1 + restrictText + ident;

    s.getRewriter().ReplaceText({getAttrFullSourceRange(a).getBegin(), decl.getEndLoc()},
                                part1 + restrictText + ident);

    return {};
}

}  // namespace oklt::cuda_subset
