#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/implicid_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleGlobalConstant(oklt::SessionStage& s, const clang::VarDecl& decl) {
    const std::string CUDA_CONST_QUALIFIER = "__constant__";
    return oklt::handleGlobalConstant(s, decl, CUDA_CONST_QUALIFIER);
}

__attribute__((constructor)) void registeCUDAGlobalConstantHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(TargetBackend::CUDA,
                                                                         handleGlobalConstant);

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register implicit handler for global constant");
    }
}
}  // namespace
