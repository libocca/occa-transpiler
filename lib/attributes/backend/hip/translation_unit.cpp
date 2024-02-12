#include <clang/AST/Decl.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace {
using namespace oklt;
using namespace clang;

bool handleTranslationUnit(const clang::Decl* d, SessionStage& s) {
  // #ifdef TRANSPILER_DEBUG_LOG
  llvm::outs() << "[DEBUG] Found translation unit\n";
// #endif

  return true;
}

__attribute__((constructor)) void registerKernelHandler() {
  auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
    {TargetBackend::HIP, clang::Decl::Kind::TranslationUnit}, DeclHandler{handleTranslationUnit});

  if (!ok) {
    llvm::errs() << "failed to register implicit handler for global constant\n";
  }
}
}  // namespace
