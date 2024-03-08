#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

namespace {
using namespace oklt;
using namespace clang;

const std::string exclusiveNullText = "_occa_exclusive_index = 0;\n";
const std::string exclusiveIncText = "++_occa_exclusive_index;\n";

HandleResult handleOPNMPInnerAttribute(const Attr& a,
                                       const ForStmt& stmt,
                                       const AttributedLoop* params,
                                       SessionStage& s) {
#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "handle attribute: " << a.getNormalizedFullName() << '\n';
#endif
    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@inner params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@inner: failed to fetch loop meta data from sema"});
    }

    auto& rewriter = s.getRewriter();

    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.RemoveText(attrRange);

    // Top most `@inner` loop
    auto parent = loopInfo->getAttributedParent();
    if (parent && parent->metadata.isOuter()) {
        // Get `@outer` attributed loop
        auto outerParent = parent;
        while (outerParent && !outerParent->metadata.isOuter()) {
            outerParent = outerParent->parent;
        }

        if (outerParent && !outerParent->vars.exclusive.empty()) {
            rewriter.InsertTextBefore(stmt.getBeginLoc(), exclusiveNullText);
        }
    }

    // Bottom most `@inner` loop
    if (loopInfo->children.empty()) {
        // Get `@outer` attributed loop
        auto outerParent = parent;
        while (outerParent && !outerParent->metadata.isOuter()) {
            outerParent = outerParent->parent;
        }

        if (outerParent && !outerParent->vars.exclusive.empty()) {
            auto compStmt = dyn_cast_or_null<CompoundStmt>(loopInfo->stmt.getBody());
            SourceLocation incLoc =
                compStmt ? compStmt->getRBracLoc().getLocWithOffset(-1) : stmt.getEndLoc();
            rewriter.InsertTextBefore(incLoc, exclusiveIncText);
        }
    }

    return {};
}

__attribute__((constructor)) void registerOPENMPOuterHandler() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::OPENMP, INNER_ATTR_NAME},
        makeSpecificAttrHandle(handleOPNMPInnerAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << EXCLUSIVE_ATTR_NAME
                     << " attribute handler (OpenMP)\n";
    }
}
}  // namespace
