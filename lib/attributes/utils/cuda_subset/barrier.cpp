#include "attributes/utils/cuda_subset/common.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "pipeline/stages/transpiler/error_codes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

namespace oklt::cuda_subset {

oklt::HandleResult handleBarrierAttribute(const clang::Attr& attr,
                                          const clang::Stmt& stmt,
                                          const oklt::AttributedBarrier* params,
                                          SessionStage& stage) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << attr.getNormalizedFullName() << '\n';
#endif
    if (!params) {
        return tl::make_unexpected(
            makeError(OkltTranspilerErrorCode::INTERNAL_ERROR_PARAMS_NULL_OBJ,
                      "params is null object in handleBarrierAttribute"));
    }
    std::string replacement = cuda_subset::SYNC_THREADS_BARRIER;
    if (params->type == BarrierType::syncWarp) {
        replacement = "__syncwarp()";
    }
    auto range = getAttrFullSourceRange(attr);
    stage.getRewriter().ReplaceText(range, replacement);

    return {};
}
}  // namespace oklt::cuda_subset
