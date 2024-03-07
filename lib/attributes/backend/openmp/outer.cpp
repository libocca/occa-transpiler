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

const std::string prefixText = "#pragma omp parallel for\n";

HandleResult handleOPENMPOuterAttribute(const Attr& a,
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

    auto& rewriter = s.getRewriter();
    auto opts = Rewriter::RewriteOptions();
    opts.RemoveLineIfEmpty = true;

    auto attrRange = getAttrFullSourceRange(a);
    rewriter.RemoveText(attrRange, opts);

    auto parent = loopInfo->getAttributedParent();
    if (!parent) {
        rewriter.InsertText(attrRange.getBegin(), prefixText, false, true);
    }

    return {};
}

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, OUTER_ATTR_NAME},
        makeSpecificAttrHandle(handleOPENMPOuterAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << OUTER_ATTR_NAME << " attribute handler (OpenMP)\n";
    }
}
}  // namespace