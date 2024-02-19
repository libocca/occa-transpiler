#include <clang/Lex/Lexer.h>
#include <oklt/core/ast_processors/okl_sema_processor/okl_sema_ctx.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/core/utils/attributes.h>

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
    auto kernelInfo = s.tryEmplaceUserCtx<OklSemaCtx>().getParsingKernelInfo();
    if (!kernelInfo) {
        // INFO: internal error
        return false;
    }
    auto& ctx = varDecl->getASTContext();
    auto& sm = ctx.getSourceManager();
    auto& opts = ctx.getLangOpts();
    SourceRange r1(varDecl->getSourceRange().getBegin(), identifierLoc);
    auto part1 = clang::Lexer::getSourceText(CharSourceRange::getCharRange(r1), sm, opts).str();
    auto ident = varDecl->getQualifiedNameAsString();
    std::string modifiedArgument = part1 + " " + restrictText + " " + ident;
    kernelInfo->argStrs.push_back(modifiedArgument);
    return true;
}

__attribute__((constructor)) void registerRestrictHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::CUDA, RESTRICT_ATTR_NAME}, AttrDeclHandler{handleCUDARestrictAttribute});

    if (!ok) {
        llvm::errs() << "failed to register " << RESTRICT_ATTR_NAME << " attribute handler\n";
    }
}
}  // namespace
