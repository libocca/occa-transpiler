#include "attributes/utils/cuda_subset/common.h"
#include "attributes/utils/cuda_subset/handle.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "pipeline/core/error_codes.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include <spdlog/spdlog.h>

namespace oklt::cuda_subset {

oklt::HandleResult handleBarrierAttribute(SessionStage& stage,
                                          const clang::Stmt& stmt,
                                          const clang::Attr& attr,
                                          const oklt::AttributedBarrier* params) {
    SPDLOG_DEBUG("Handle [@barrier] attribute");

    if (!params) {
        return tl::make_unexpected(
            makeError(OkltPipelineErrorCode::INTERNAL_ERROR_PARAMS_NULL_OBJ,
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
