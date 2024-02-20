#include <clang/AST/Decl.h>
#include <oklt/util/string_utils.h>
#include <functional>
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "attributes/utils/loop_meta_data.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "handle.h"

namespace oklt::cuda_subset {
using namespace clang;
bool handleOuterAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @outer attribute\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
