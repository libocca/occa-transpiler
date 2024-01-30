#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>

namespace {
using namespace oklt;

bool parseDimAttribute(const clang::Attr* a, SessionStage&) {
  llvm::outs() << "parse attribute: " << a->getNormalizedFullName() << '\n';
  return true;
}

bool handleDimDeclAttrbute(const clang::Attr* a, const clang::Decl* decl, SessionStage& s) {
  llvm::outs() << "handle decl attribute: " << a->getNormalizedFullName() << '\n';
  return true;
}

bool handleDimStmtAttrbute(const clang::Attr* a, const clang::Stmt* stmt, SessionStage& s) {
  llvm::outs() << "handle stmt attribute: " << a->getNormalizedFullName() << '\n';
  return true;
}

__attribute__((constructor)) void registerDimHandler() {
  auto ok = oklt::AttributeManager::instance().registerCommonHandler(
              DIM_ATTR_NAME, {parseDimAttribute, handleDimDeclAttrbute}) &&
            oklt::AttributeManager::instance().registerCommonHandler(
              DIM_ATTR_NAME, {parseDimAttribute, handleDimStmtAttrbute});

  if (!ok) {
    llvm::errs() << "failed to register " << DIM_ATTR_NAME << " attribute handler\n";
  }
}
}  // namespace
