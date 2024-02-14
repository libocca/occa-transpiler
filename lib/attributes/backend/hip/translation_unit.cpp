#include <clang/AST/Decl.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace {
using namespace oklt;
using namespace clang;

bool handleTranslationUnit(const clang::Decl* decl, SessionStage& s) {
    if (!isa<TranslationUnitDecl>(decl)) {
        return true;
    }

    auto& sourceManager = s.getCompiler().getSourceManager();
    auto mainFileId = sourceManager.getMainFileID();
    auto loc = sourceManager.getLocForStartOfFile(mainFileId);
    auto& rewriter = s.getRewriter();
    rewriter.InsertTextBefore(loc, "#include <hip/hip_runtime.h>\n");

#ifdef TRANSPILER_DEBUG_LOG
    auto offset = sourceManager.getFileOffset(decl->getLocation());
    llvm::outs() << "[DEBUG] Found translation unit, offset: " << offset << "\n";
#endif

    return true;
}

__attribute__((constructor)) void registerKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::TranslationUnit},
        DeclHandler{handleTranslationUnit});

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for translation unit (HIP)\n";
    }
}
}  // namespace
