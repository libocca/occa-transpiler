#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleTranslationUnitHip(const clang::TranslationUnitDecl& decl, SessionStage& s) {
    const std::string HIP_INCLUDE = "#include <hip/hip_runtime.h>";
    return oklt::handleTranslationUnit(decl, s, HIP_INCLUDE);
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::TranslationUnit},
        makeSpecificImplicitHandle(handleTranslationUnitHip));

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for translation unit (HIP)\n";
    }
}
}  // namespace
