#include <oklt/util/string_utils.h>

#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>
#include <any>

namespace oklt::cuda_subset {
using namespace clang;
tl::expected<std::any, Error> handleInnerAttribute(const clang::Attr* a,
                                                   const clang::Stmt* d,
                                                   const std::any& params,
                                                   SessionStage& s) {
    auto loopParams = std::any_cast<AttributedLoop>(params);

    auto& astCtx = s.getCompiler().getASTContext();
    if (!isa<ForStmt>(d)) {
        s.pushError(std::error_code(), "@inner can be applied to only for loop");
        return false;
    }
    const auto* forStmt = dyn_cast<ForStmt>(d);
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto forLoopMetaData = sema.getLoopMetaData(forStmt);
    if (!forLoopMetaData) {
        s.pushError(std::error_code(), "@tile: failed to fetch loop meta data from sema");
        return false;
    }

    int openedScopeCounter = 0;
    auto prefixCode = inner_outer::buildInnerOuterLoopIdxLine(
        forLoopMetaData.value(), loopParams, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @inner attribute\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
