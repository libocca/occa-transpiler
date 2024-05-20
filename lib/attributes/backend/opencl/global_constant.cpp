#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/implicid_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleGlobalConstant(oklt::SessionStage& s, const clang::VarDecl& decl) {
    const std::string OPENCL_CONST_QUALIFIER = "__constant";
    return oklt::handleGlobalConstant(s, decl, OPENCL_CONST_QUALIFIER);
}

__attribute__((constructor)) void registeCUDAGlobalConstantHandler() {
    auto ok = registerImplicitHandler(TargetBackend::OPENCL, handleGlobalConstant);

    if (!ok) {
        SPDLOG_ERROR("[OPENCL] Failed to register implicit handler for global constant");
    }
}
}  // namespace
