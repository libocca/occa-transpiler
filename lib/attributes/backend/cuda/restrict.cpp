#include "attributes/attribute_names.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/Lex/Lexer.h>

namespace {
using namespace oklt;
using namespace clang;

bool handleCUDARestrictAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
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
    std::string restrictText = " __restrict__ ";
    rewriter.InsertText(identifierLoc, restrictText, false, false);

    auto& ctx = varDecl->getASTContext();
    auto& sm = ctx.getSourceManager();
    auto& opts = ctx.getLangOpts();
    SourceRange r1(varDecl->getSourceRange().getBegin(), identifierLoc);
    auto part1 = clang::Lexer::getSourceText(CharSourceRange::getCharRange(r1), sm, opts).str();
    auto ident = varDecl->getQualifiedNameAsString();
    std::string modifiedArgument = part1 + " " + restrictText + " " + ident;

    // set transpiled arg attr modifier string
    if (s.getAstProccesorType() == AstProcessorType::OKL_WITH_SEMA) {
        s.tryEmplaceUserCtx<OklSemaCtx>().setKernelArgRawString(dyn_cast<ParmVarDecl>(d),
                                                                modifiedArgument);
    }

    return true;
}

__attribute__((constructor)) void registerCUDARestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, RESTRICT_ATTR_NAME}, AttrDeclHandler{handleCUDARestrictAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
