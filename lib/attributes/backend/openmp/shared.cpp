#include "attributes/backend/openmp/common.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleOPENMPSharedAttribute(const Attr& a, const Decl& decl, SessionStage& s) {
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

    auto& loopInfoEx = openmp::getBackendCtxFromStage(s).getLoopInfo(loopInfo);

    // Process later when processing ForStmt
    loopInfoEx.shared.emplace_back(std::ref(decl));

    SourceRange attr_range = getAttrFullSourceRange(a);
    s.getRewriter().RemoveText(attr_range);

    return {};
}

__attribute__((constructor)) void registerOPENMPSharedHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, SHARED_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPSharedAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << SHARED_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
