#include "attributes/attribute_names.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"
#include "core/transpilation.h"
#include "core/transpilation_encoded_names.h"

namespace {
using namespace oklt;
using namespace clang;
const std::string RESTRICT_MODIFIER = "__restrict__";

HandleResult handleRestrictAttribute(const clang::Attr& a,
                                     const clang::ParmVarDecl& parmDecl,
                                     SessionStage& s) {
    SourceLocation identifierLoc = parmDecl.getLocation();
    std::string restrictText = " " + RESTRICT_MODIFIER + " ";

    // INFO: might be better to use rewriter.getRewrittenText() method

    auto& ctx = parmDecl.getASTContext();
    SourceRange r1(parmDecl.getSourceRange().getBegin(), identifierLoc);
    auto part1 = getSourceText(r1, ctx);
    auto ident = parmDecl.getQualifiedNameAsString();
    std::string modifiedArgument = part1 + restrictText + ident;

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @restrict.\n";
#endif

    return TranspilationBuilder(s.getCompiler().getSourceManager(), a.getNormalizedFullName(), 1)
        .addReplacement(OKL_TRANSPILED_ARG,
                        getAttrFullSourceRange(a).getBegin(),
                        parmDecl.getEndLoc(),
                        part1 + restrictText + ident)
        .build();

}

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, RESTRICT_ATTR_NAME},
        makeSpecificAttrHandle(handleRestrictAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
