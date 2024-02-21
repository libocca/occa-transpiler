#include <clang/AST/Decl.h>
#include <oklt/util/string_utils.h>
#include <functional>
#include "attributes/frontend/params/tile.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "attributes/utils/loop_meta_data.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "handle.h"

namespace oklt::cuda_subset {
using namespace clang;
bool handleInnerAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
    auto usrCtxKey = util::pointerToStr(static_cast<const void*>(a));
    auto loopParams = std::any_cast<AttributedLoop>(s.getUserCtx(usrCtxKey));
    if (loopParams == nullptr) {
        s.pushError(std::error_code(), "No @inner params in user context");
        return false;
    }

    auto& astCtx = s.getCompiler().getASTContext();
    if (!isa<ForStmt>(d)) {
        s.pushError(std::error_code(), "@inner can be applied to only for loop");
        return false;
    }
    const auto* forStmt = dyn_cast<ForStmt>(d);
    auto forLoop = ParseForStmt(const_cast<ForStmt*>(forStmt), astCtx);
    int openedScopeCounter = 0;
    auto prefixCode =
        inner_outer::buildInnerOuterLoopIdxLine(forLoop, *loopParams, openedScopeCounter);
    auto suffixCode = buildCloseScopes(openedScopeCounter);

    replaceAttributedLoop(a, forStmt, prefixCode, suffixCode, s);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @inner attribute\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
