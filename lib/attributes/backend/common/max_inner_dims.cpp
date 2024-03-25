#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"

namespace {
using namespace oklt;
using namespace clang;

HandleResult handleMaxInnerDimsStmtAttribute(const clang::Attr& a,
                                             const clang::ForStmt& forStmt,
                                             const AttributedLoopInnerSize* params,
                                             SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif

    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@max_inner_dims params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(forStmt);
    if (loopInfo && !loopInfo->parent && !params->size.empty()) {
        OklLoopInfo::OptSizes sz = {1, 1, 1};
        for (size_t i = 0; i < params->size.size(); ++i) {
            auto s = params->size[i];
            if (s > 0) {
                sz[i] = s;
            }
        }

        loopInfo->overridenInnerSizes = sz;
    }

    removeAttribute(a, s);
    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerCommonHandler(
        MAX_INNER_DIMS, makeSpecificAttrHandle(handleMaxInnerDimsStmtAttribute));
    if (!ok) {
        llvm::errs() << "failed to register " << MAX_INNER_DIMS << " attribute decl handler\n";
    }
}
}  // namespace
