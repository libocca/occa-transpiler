#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string CUDA_FUNCTION_QUALIFIER = "__device__";

HandleResult handleCudaGlobalFunction(SessionStage& s, const clang::FunctionDecl& d) {
    return handleGlobalFunction(s, d, CUDA_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, ASTNodeKind::getFromNodeKind<FunctionDecl>()},
        makeSpecificImplicitHandle(handleCudaGlobalFunction));

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register implicit handler for global function");
    }
}
}  // namespace
