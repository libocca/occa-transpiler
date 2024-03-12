#include "attributes/frontend/params/barrier.h"
#include "attributes/utils/serial_subset/common.h"

namespace oklt::serial_subset {
using namespace clang;

HandleResult handleEmptyStmtAttribute(const Attr& a,
                                      const Stmt& stmt,
                                      const std::any* params,
                                      SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    removeAttribute(a, s);
    return {};
}

HandleResult handleEmptyDeclAttribute(const Attr& a,
                                      const Decl& decl,
                                      const std::any* params,
                                      SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    removeAttribute(a, s);
    return {};
}

}  // namespace oklt::serial_subset
