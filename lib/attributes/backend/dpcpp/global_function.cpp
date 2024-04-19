#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/implicid_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleGlobalFunctionDpcpp(oklt::SessionStage& s, const clang::FunctionDecl& decl) {
    const std::string HIP_FUNCTION_QUALIFIER = "SYCL_EXTERNAL";
    return oklt::handleGlobalFunction(s, decl, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerTranslationUnitAttrBackend() {
    auto ok =
        HandlerManager::registerImplicitHandler(TargetBackend::DPCPP, handleGlobalFunctionDpcpp);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register implicit handler for global function");
    }
}
}  // namespace
