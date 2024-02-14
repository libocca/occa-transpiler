#include "attributes/backend/common/cuda_like_backends/cuda_like.h"
#include <oklt/core/attribute_manager/attribute_manager.h>

namespace {
using namespace oklt;
__attribute__((constructor)) void registerKernelHandler() {
  auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
    {TargetBackend::HIP, clang::Decl::Kind::Var}, DeclHandler{cuda_like::handleGlobalConstant});

  if (!ok) {
    llvm::errs() << "Failed to register implicit handler for global constant (HIP)\n";
  }
}
}  // namespace
