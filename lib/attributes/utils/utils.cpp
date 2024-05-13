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

std::string getCleanTypeString(clang::QualType t) {
    static std::string annoTypeStr = " [[clang::annotate_type(...)]]";
    auto str = t.getAsString();

    auto pos = str.find(annoTypeStr);
    while (pos != std::string::npos) {
        str.replace(pos, annoTypeStr.size(), "");
        pos = str.find(annoTypeStr);
    }

    return str;
}

}  // namespace oklt
