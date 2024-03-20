#include <oklt/core/kernel_metadata.h>
#include <oklt/util/string_utils.h>

#include "attributes/frontend/params/loop.h"
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/common.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"

#include "core/attribute_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "tl/expected.hpp"

#include <clang/AST/Decl.h>
#include <clang/AST/Stmt.h>

namespace oklt::cuda_subset {
using namespace clang;

HandleResult handleInnerAttribute(const clang::Attr& a,
                                  const clang::ForStmt& forStmt,
                                  const AttributedLoop* params,
                                  SessionStage& s) {
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{std::error_code(), "@inner: failed to fetch loop meta data from sema"});
    }

    // Auto Axes in loopInfo are replaced with specific. TODO: maybe somehow update params earlier?
    auto updatedParams = *params;
    updatedParams.axis = loopInfo->axis.front();

    int openedScopeCounter = 0;
    auto prefixCode = inner_outer::buildInnerOuterLoopIdxLine(
        *loopInfo, updatedParams, openedScopeCounter, s.getRewriter());
    auto suffixCode = buildCloseScopes(openedScopeCounter);
    if (loopInfo->shouldSync()) {
        suffixCode += cuda_subset::SYNC_THREADS_BARRIER + ";\n";
    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @inner attribute\n";
#endif

    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s, true);
}
}  // namespace oklt::cuda_subset
