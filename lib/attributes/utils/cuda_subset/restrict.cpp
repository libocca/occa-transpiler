#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/result.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"
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
                                     const clang::ParmVarDecl& parmDecl,
                                     SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    SourceLocation identifierLoc = parmDecl.getLocation();
    std::string restrictText = " " + RESTRICT_MODIFIER + " ";
    // INFO: might be better to use rewriter.getRewrittenText() method

    auto& ctx = parmDecl.getASTContext();
    SourceRange r1(parmDecl.getSourceRange().getBegin(), identifierLoc);
    auto part1 = getSourceText(r1, ctx);
    auto ident = parmDecl.getQualifiedNameAsString();
    std::string modifiedArgument = part1 + restrictText + ident;

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
        .addReplacement(OKL_TRANSPILED_ARG,
                        getAttrFullSourceRange(a).getBegin(),
                        parmDecl.getEndLoc(),
                        part1 + restrictText + ident)
        .build();
}

}  // namespace oklt::cuda_subset
