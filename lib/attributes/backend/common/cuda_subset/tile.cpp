#include <clang/AST/Decl.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>

namespace oklt::cuda_subset {

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
  llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
  return true;
}
}  // namespace oklt::cuda_subset
