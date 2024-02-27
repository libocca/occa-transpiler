#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleTranslationUnit(const clang::TranslationUnitDecl& decl, SessionStage& s) {
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

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::TranslationUnit},
        makeSpecificImplicitHandle(handleTranslationUnit));

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for translation unit (HIP)\n";
    }
}
}  // namespace
