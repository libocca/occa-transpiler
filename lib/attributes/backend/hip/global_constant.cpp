#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;

HandleResult handleHIPGlobalConstant(const clang::VarDecl& decl, oklt::SessionStage& s) {
    const std::string HIP_CONST_QUALIFIER = "__constant__";
    return oklt::handleGlobalConstant(decl, s, HIP_CONST_QUALIFIER);
}

__attribute__((constructor)) void registeHIPGlobalConstantHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::Var},
        makeSpecificImplicitHandle(handleHIPGlobalConstant));

    if (!ok) {
        SPDLOG_ERROR("[HIP] Failed to register implicit handler for global constant");
    }
}
}  // namespace
