#include <oklt/util/string_utils.h>

#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"

#include "core/attribute_manager/result.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>

namespace oklt::cuda_subset {
using namespace clang;
HandleResult handleInnerAttribute(const clang::Attr& a,
                                  const clang::ForStmt& forStmt,
                                  const AttributedLoop* params,
                                  SessionStage& s) {
    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{std::error_code(), "@inner: failed to fetch loop meta data from sema"});
    }

    int openedScopeCounter = 0;
    auto prefixCode = inner_outer::buildInnerOuterLoopIdxLine(loopInfo.value(), *params, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @inner attribute\n";
#endif
    return replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);
}
}  // namespace oklt::cuda_subset
