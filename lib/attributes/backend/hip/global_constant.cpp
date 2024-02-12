#include <clang/AST/Decl.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace {
using namespace oklt;
using namespace clang;

bool handleGlobalConstant(const clang::Decl* d, SessionStage& s) {
  // Should be variable declaration
  if (!isa<VarDecl>(d)) {
    return true;
  }

  auto var = dyn_cast<VarDecl>(d);

  // Should be global variable
  if (var->isLocalVarDecl()) {
    return true;
  }

  // Should be constant qualified
  if (!var->getType().isConstQualified()) {
    return true;
  }

#ifdef TRANSPILER_DEBUG_LOG
  auto type_str = var->getType().getAsString();
  auto declname = var->getDeclName().getAsString();

  llvm::outs() << "[DEBUG] Found constant global variable declaration: type: " << type_str
               << ", name: " << declname << "\n";
#endif

  auto& rewriter = s.getRewriter();
  auto loc = var->getSourceRange().getBegin();
  rewriter.ReplaceText(loc, 5, "__constant__"); // Replace 'const' with __constant__

  return true;
}

__attribute__((constructor)) void registerKernelHandler() {
  auto ok = oklt::AttributeManager::instance().registerImplicitHandler(
    {TargetBackend::HIP, clang::Decl::Kind::Var}, DeclHandler{handleGlobalConstant});

  if (!ok) {
    llvm::errs() << "failed to register implicit handler for global constant\n";
  }
}
}  // namespace
