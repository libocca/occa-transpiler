#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

const std::string CUDA_FUNCTION_QUALIFIER = "__device__";

HandleResult handleCudaGlobalFunction(const clang::FunctionDecl& d, SessionStage& s) {
    return handleGlobalFunction(d, s, CUDA_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::Function},
        makeSpecificImplicitHandle(handleCudaGlobalFunction));

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for global function (CUDA)\n";
    }
}
}  // namespace
