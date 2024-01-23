#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool parseSharedAttribute(const clang::Attr*, SessionStage&) {
  llvm::outs() << "<<<parse shared attr for cuda>>>\n";
  return true;
}

bool handleSharedAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
  llvm::outs() << "<<<handle shared attr for cuda>>>\n";
  return true;
}

__attribute__((constructor)) void registerSharedHandler() {
  auto ok = oklt::AttributeManager::instance().registerBackendHandler(
    {TRANSPILER_TYPE::CUDA, SHARED_ATTR_NAME}, {parseSharedAttribute, handleSharedAttribute});

  if (!ok) {
    llvm::errs() << "failed to register " << SHARED_ATTR_NAME << " attribute handler\n";
  }
}
}  // namespace
