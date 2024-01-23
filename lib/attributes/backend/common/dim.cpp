#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool parseDimAttribute(const clang::Attr*, SessionStage&) {
  llvm::outs() << "<<<parse dim attr for cuda>>>\n";
  return true;
}

bool handleDimAttrbute(const clang::Attr* a, const clang::Decl* d, SessionStage& s) {
  llvm::outs() << "<<<handle dim attr for cuda>>>\n";
  return true;
}

__attribute__((constructor)) void registerDimHandler() {
  oklt::AttributeManager::instance().registerCommonHandler(DIM_ATTR_NAME,
                                                           {parseDimAttribute, handleDimAttrbute});
}
}  // namespace
