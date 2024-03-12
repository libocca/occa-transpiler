#include "attributes/utils/serial_subset/common.h"

namespace oklt::serial_subset {
using namespace clang;

HandleResult handleOuterAttribute(const Attr& a,
                                  const ForStmt& stmt,
                                  const AttributedLoop* params,
                                  SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@outer params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@outer: failed to fetch loop meta data from sema"});
    }

    removeAttribute(a, s);
    return {};
}

}  // namespace oklt::serial_subset
