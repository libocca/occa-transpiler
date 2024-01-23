#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool parseKernelAttribute(const clang::Attr*, SessionStage&) {
  llvm::outs() << "<<<parse kernel attr for cuda>>>\n";
  return true;
}

bool handleKernelAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
  llvm::outs() << "<<<handle kernel attr for cuda>>>\n";
  return true;
}

__attribute__((constructor)) void registerKernelHandler() {
  auto ok = oklt::AttributeManager::instance().registerBackendHandler(
    {TRANSPILER_TYPE::CUDA, KERNEL_ATTR_NAME}, {parseKernelAttribute, handleKernelAttribute});

  if (!ok) {
    llvm::errs() << "failed to register " << KERNEL_ATTR_NAME << " attribute handler\n";
  }
}
}  // namespace
