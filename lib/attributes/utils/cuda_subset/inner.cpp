#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/common.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "attributes/utils/kernel_utils.h"

#include "core/handler_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "tl/expected.hpp"

#include <clang/AST/Stmt.h>

#include <spdlog/spdlog.h>

namespace oklt::cuda_subset {
using namespace clang;

HandleResult handleInnerAttribute(SessionStage& s,
                                  const clang::ForStmt& forStmt,
                                  const clang::Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@inner] attribute");
    handleChildAttr(s, forStmt, NO_BARRIER_ATTR_NAME);

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{std::error_code(), "@inner: failed to fetch loop meta data from sema"});
    }

    auto updatedParams = *params;
    // Auto Axis in loopInfo are replaced with specific. TODO: maybe somehow update params earlier?
    updatedParams.axis = loopInfo->axis.front();

    std::string afterRBraceCode = "";
    if (loopInfo->shouldSync()) {
        afterRBraceCode += cuda_subset::SYNC_THREADS_BARRIER + ";\n";
    }

    int openedScopeCounter = 0;
    auto prefixCode = inner_outer::buildInnerOuterLoopIdxLine(
        *loopInfo, updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    return replaceAttributedLoop(s, forStmt, a, suffixCode, afterRBraceCode, prefixCode, true);
}
}  // namespace oklt::cuda_subset
