#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string CUDA_RT_INC = "<cuda_runtime.h>";
__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::TranslationUnit},
        DeclHandler{
            [](const auto& d, auto& s) { return handleTranslationUnit(d, s, CUDA_RT_INC); }});

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for translation unit (CUDA)\n";
    }
}
}  // namespace
