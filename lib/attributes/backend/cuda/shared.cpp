#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool parse_shared_attr(const clang::Attr*, SessionStage&) {
  llvm::outs() << "<<<parse shared attr for cuda>>>\n";
  return true;
}

bool handle_shared_attr(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
  llvm::outs() << "<<<handle shared attr for cuda>>>\n";
  return true;
}

__attribute__((constructor)) void register_shared_handler() {
  oklt::AttributeManager::instance().registerBackendHandler(
      BackendAttributeMap::KeyType{TRANSPILER_TYPE::CUDA, SHARED_ATTR_NAME},
      AttrDeclHandler{parse_shared_attr, handle_shared_attr});
}
}  // namespace
