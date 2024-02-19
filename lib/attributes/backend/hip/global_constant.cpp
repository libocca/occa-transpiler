#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>

namespace {
using namespace oklt;
__attribute__((constructor)) void registeGlovalConstantHandler() {
    auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
        {TargetBackend::HIP, clang::Decl::Kind::Var},
        DeclHandler{cuda_subset::handleGlobalConstant});

    if (!ok) {
        llvm::errs() << "Failed to register implicit handler for global constant (HIP)\n";
    }
}
}  // namespace
