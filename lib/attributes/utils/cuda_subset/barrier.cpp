#include "core/ast_processors/okl_sema_processor/okl_sema_ctx.h"
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

#include <clang/AST/Attr.h>

namespace oklt::cuda_subset {
using namespace clang;
HandleResult handleBarrierAttribute(const clang::Attr& a,
                                     const clang::Stmt& stmt,
                                     SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    auto& rewriter = s.getRewriter();
    SourceRange range(a.getRange().getBegin().getLocWithOffset(-2), stmt.getEndLoc());
    rewriter.ReplaceText(range, "__syncthreads();");

    return true;
}

}  // namespace oklt::cuda_subset
