#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>

namespace {
using namespace oklt;
__attribute__((constructor)) void registerGlobalFunctionHandler() {
  auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
    {TargetBackend::HIP, clang::Decl::Kind::Function},
    DeclHandler{cuda_subset::handleGlobalFunction});

  if (!ok) {
    llvm::errs() << "Failed to register implicit handler for global function (HIP)\n";
  }
}
}  // namespace
