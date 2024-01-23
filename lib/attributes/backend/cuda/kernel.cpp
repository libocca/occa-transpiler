#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool parse_kernel_attr(const clang::Attr*, SessionStage&) {
  llvm::outs() << "<<<parse kernel attr for cuda>>>\n";
  return true;
}

bool handle_kernel_attr(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
  llvm::outs() << "<<<handle kernel attr for cuda>>>\n";
  return true;
}

__attribute__((constructor)) void register_kernel_handler() {
  oklt::AttributeManager::instance().registerBackendHandler(
      BackendAttributeMap::KeyType{TRANSPILER_TYPE::CUDA, KERNEL_ATTR_NAME},
      AttrDeclHandler{parse_kernel_attr, handle_kernel_attr});
}
}  // namespace
