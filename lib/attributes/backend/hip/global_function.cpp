#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

HandleResult handleHIPGlobalFunction(oklt::SessionStage& s, const clang::FunctionDecl& decl) {
    const std::string HIP_FUNCTION_QUALIFIER = "__device__";
    return oklt::handleGlobalFunction(s, decl, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerHIPKernelHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::Function},
        makeSpecificImplicitHandle(handleHIPGlobalFunction));

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register implicit handler for global function");
    }
}
}  // namespace
