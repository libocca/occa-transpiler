#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleTranslationUnitCuda(const clang::Decl* decl, SessionStage& s) {
    const std::string CUDA_INCLUDE = "#include <cuda_runtime.h>";
    return oklt::handleTranslationUnit(decl, s, CUDA_INCLUDE);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::TranslationUnit},
        DeclHandler{handleTranslationUnitCuda});

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for translation unit (HIP)\n";
    }
}
}  // namespace
