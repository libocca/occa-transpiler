#include <oklt/util/string_utils.h>
#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Decl.h>

namespace oklt::cuda_subset {
using namespace clang;
// TODO: function is very similar to handleInnerAttribute
HandleResult handleOuterAttribute(const clang::Attr& a,
                                  const clang::ForStmt& forStmt,
                                  const AttributedLoop* params,
                                  SessionStage& s) {
    auto& astCtx = s.getCompiler().getASTContext();
    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto forLoopMetaData = sema.getLoopMetaData(forStmt);
    if (!forLoopMetaData) {
        s.pushError(std::error_code(), "@tile: failed to fetch loop meta data from sema");
        return false;
    }

    int openedScopeCounter = 0;
    auto prefixCode = inner_outer::buildInnerOuterLoopIdxLine(
        forLoopMetaData.value(), *params, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @outer attribute\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
