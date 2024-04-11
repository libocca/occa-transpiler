#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleGlobalFunctionDpcpp(oklt::SessionStage& s, const clang::FunctionDecl& decl) {
    const std::string HIP_FUNCTION_QUALIFIER = "SYCL_EXTERNAL";
    return oklt::handleGlobalFunction(s, decl, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::DPCPP, ASTNodeKind::getFromNodeKind<FunctionDecl>()},
        makeSpecificImplicitHandle(handleGlobalFunctionDpcpp));

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register implicit handler for global function");
    }
}
}  // namespace
