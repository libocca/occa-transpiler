#include <oklt/core/attribute_manager/attribute_manager.h>
#include "attributes/utils/replace_attribute.h"

namespace {
using namespace oklt;

bool handleGlobalConstant(const clang::Decl* decl, oklt::SessionStage& s) {
  const std::string HIP_CONST_QUALIFIER = "__constant__";
  return oklt::handleGlobalConstant(decl, s, HIP_CONST_QUALIFIER);
}

__attribute__((constructor)) void registerKernelHandler() {
  auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
    {TargetBackend::HIP, clang::Decl::Kind::Var}, DeclHandler{handleGlobalConstant});

  if (!ok) {
    llvm::errs() << "Failed to register implicit handler for global constant (HIP)\n";
  }
}
}  // namespace
