#include "attributes/utils/utils.h"

#include <clang/AST/ParentMapContext.h>

namespace oklt {

const clang::AttributedStmt* getAttributedStmt(SessionStage& s, const clang::Stmt& stmt) {
    auto& ctx = s.getCompiler().getASTContext();
    const auto parents = ctx.getParentMapContext().getParents(stmt);
    if (parents.empty())
        return nullptr;

    return parents[0].get<clang::AttributedStmt>();
}
}  // namespace oklt
