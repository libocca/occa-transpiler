#include <oklt/util/string_utils.h>

#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"

#include "core/handler_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>

#include <spdlog/spdlog.h>

namespace oklt::cuda_subset {
using namespace clang;

HandleResult handleOuterAttribute(SessionStage& s,
                                  const clang::ForStmt& forStmt,
                                  const clang::Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@outer] attribute");

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{
            .ec = std::error_code(), .desc = "@outer: failed to fetch loop meta data from sema"});
    }

    auto updatedParams = *params;
    // Auto Axis in loopInfo are replaced with specific. TODO: maybe somehow update params earlier?
    updatedParams.axis = loopInfo->axis.front();

    int openedScopeCounter = 0;
    auto prefixCode = inner_outer::buildInnerOuterLoopIdxLine(
        *loopInfo, updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    return replaceAttributedLoop(s, forStmt, a, suffixCode, prefixCode, true);
}

}  // namespace oklt::cuda_subset
