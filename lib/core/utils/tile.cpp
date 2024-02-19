#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "attributes/backend/common/cuda_subset/cuda_subset.h"

#include <clang/AST/Decl.h>
namespace oklt::cuda_subset {

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
    llvm::outs() << "handle attribute: " << a->getNormalizedFullName() << '\n';
    return true;
}
}  // namespace oklt::cuda_subset
