#include "attributes/utils/replace_attribute.h"
#include "core/handler_manager/handler_manager.h"
#include "core/handler_manager/implicid_handler.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

const std::string CUDA_FUNCTION_QUALIFIER = "__device__";

HandleResult handleCudaGlobalFunction(SessionStage& s, const clang::FunctionDecl& d) {
    return handleGlobalFunction(s, d, CUDA_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerImplicitHandler(TargetBackend::CUDA, handleCudaGlobalFunction);

    if (!ok) {
        SPDLOG_ERROR("[CUDA] Failed to register implicit handler for global function");
    }
}
}  // namespace
