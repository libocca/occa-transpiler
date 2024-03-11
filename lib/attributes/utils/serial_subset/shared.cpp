#include "attributes/utils/serial_subset/common.h"

namespace oklt::serial_subset {
using namespace clang;

HandleResult handleSharedAttribute(const Attr& a, const Decl& decl, SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@shared: failed to fetch loop meta data from sema"});
    }

    if (!loopInfo->isOuter()) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    auto child = loopInfo->getFirstAttributedChild();
    if (!child || !child->isInner()) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    auto& loopInfoEx = getBackendCtxFromStage(s).getLoopInfo(loopInfo);

    // Process later when processing ForStmt
    loopInfoEx.shared.emplace_back(std::ref(decl));

    SourceRange attr_range = getAttrFullSourceRange(a);
    s.getRewriter().RemoveText(attr_range);

    return {};
}

}  // namespace oklt::serial_subset
