#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleNoBarrierStmtAttribute(const clang::Attr& a,
                                          const clang::ForStmt& forStmt,
                                          SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@nobarrier: failed to fetch loop meta data from sema"});
    }

    loopInfo->sharedInfo.used = false;

    removeAttribute(a, s);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        NOBARRIER_ATTR_NAME, makeSpecificAttrHandle(handleNoBarrierStmtAttribute));
    if (!ok) {
        llvm::errs() << "failed to register " << NOBARRIER_ATTR_NAME << " attribute stmt handler\n";
    }
}
}  // namespace
