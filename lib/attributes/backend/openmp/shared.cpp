#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <clang/Rewrite/Core/Rewriter.h>

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

    if (loopInfo->metadata.type != LoopMetaType::Outer) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    auto child = loopInfo->getFirstAttributedChild();
    if (!child || child->metadata.type != LoopMetaType::Inner) {
        return tl::make_unexpected(
            Error{{}, "Must define [@shared] variables between [@outer] and [@inner] loops"});
    }

    SourceRange attrRange = getAttrFullSourceRange(a);
    s.getRewriter().RemoveText(attrRange);

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
