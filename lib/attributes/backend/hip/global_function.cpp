#include <oklt/core/attribute_manager/attribute_manager.h>
#include "attributes/backend/common/cuda_subset/cuda_subset.h"

namespace {
using namespace oklt;
__attribute__((constructor)) void registerKernelHandler() {
  auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
    {TargetBackend::HIP, clang::Decl::Kind::Function},
    DeclHandler{cuda_like::handleGlobalFunction});

  if (!ok) {
    llvm::errs() << "Failed to register implicit handler for global function (HIP)\n";
  }
}
}  // namespace
