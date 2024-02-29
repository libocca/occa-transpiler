#include "attributes/utils/cuda_subset/handle.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

namespace oklt::cuda_subset {
oklt::HandleResult handleBarrierAttribute(const clang::Attr& attr,
                                          const clang::Stmt& stmt,
                                          const oklt::AttributedBarrier* params,
                                          SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& rewriter = stage.getRewriter();
    removeAttribute(attr, stage);
    if (params->type == BarrierType::syncDefault) {
        rewriter.InsertText(stmt.getBeginLoc(), "__syncthreads()", false, false);
    } else {
        rewriter.InsertText(stmt.getBeginLoc(), "__syncwarp()", false, false);
    }
    return true;
}
}  // namespace oklt::cuda_subset
