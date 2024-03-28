#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

HandleResult handleGlobalFunctionDpcpp(const clang::FunctionDecl& decl, oklt::SessionStage& s) {
    const std::string HIP_FUNCTION_QUALIFIER = "SYCL_EXTERNAL";
    return oklt::handleGlobalFunction(decl, s, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::DPCPP, clang::Decl::Kind::Function},
        makeSpecificImplicitHandle(handleGlobalFunctionDpcpp));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register implicit handler for global function");
    }
}
}  // namespace
