#include "attributes/utils/replace_attribute.h"
#include "core/attribute_manager/attribute_manager.h"

namespace {
using namespace oklt;

HandleResult handleGlobalConstant(const clang::Decl* decl, oklt::SessionStage& s) {
    const std::string CUDA_CONST_QUALIFIER = "__constant__";
    return oklt::handleGlobalConstant(decl, s, CUDA_CONST_QUALIFIER);
}

__attribute__((constructor)) void registeCUDAGlobalConstantHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::Var}, DeclHandler{handleGlobalConstant});

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for global constant (CUDA)\n";
    }
}
}  // namespace
