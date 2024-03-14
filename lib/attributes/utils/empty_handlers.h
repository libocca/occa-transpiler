#include "core/attribute_manager/result.h"

#include <clang/AST/AST.h>

namespace clang {
class Attr;
class Stmt;

}  // namespace clang

namespace oklt {
class SessionStage;

inline HandleResult emptyHandleSharedStmtAttribute(const clang::Attr& a,
                                                   const clang::Stmt& stmt,
                                                   SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "Called empty stmt [@shared] handler\n";
#endif
    return {};
}

}  // namespace oklt
