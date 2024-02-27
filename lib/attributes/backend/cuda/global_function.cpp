#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

HandleResult handleCUDAGlobalFunction(const clang::FunctionDecl& decl, oklt::SessionStage& s) {
    const std::string HIP_FUNCTION_QUALIFIER = "__device__";
    return oklt::handleGlobalFunction(decl, s, HIP_FUNCTION_QUALIFIER);
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::Function},
        makeSpecificImplicitHandle(handleCUDAGlobalFunction));

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for global function (HIP)\n";
    }
}
}  // namespace
