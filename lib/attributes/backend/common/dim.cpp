#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool parse_dim_attr(const clang::Attr*, SessionStage&) {
  llvm::outs() << "<<<parse dim attr for cuda>>>\n";
  return true;
}

bool handle_dim_attr(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
  llvm::outs() << "<<<handle dim attr for cuda>>>\n";
  return true;
}

__attribute__((constructor)) void register_dim_handler() {
  oklt::AttributeManager::instance().registerCommonHandler(
      DIM_ATTR_NAME, AttrDeclHandler{parse_dim_attr, handle_dim_attr});
}
}  // namespace
