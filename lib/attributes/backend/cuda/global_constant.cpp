#include "core/attribute_manager/attribute_manager.h"

#include "attributes/backend/common/cuda_subset/cuda_subset.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registeGlobalConstantHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::CUDA, clang::Decl::Kind::Var},
        DeclHandler{cuda_subset::handleGlobalConstant});

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for global constant (HIP)\n";
    }
}
}  // namespace
