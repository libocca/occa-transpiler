#include "attributes/backend/openmp/common.h"

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

    auto& backendCtx = openmp::getBackendCtxFromStage(s);
    auto& rewriter = s.getRewriter();

    SourceRange attrRange = getAttrFullSourceRange(a);
    rewriter.RemoveText(attrRange);

    // Top most `@inner` loop
    auto parent = loopInfo->getAttributedParent();
    if (parent && parent->hasOuter()) {
        // Get `@outer` attributed loop
        auto outerParent = parent;
        while (outerParent && !outerParent->hasOuter()) {
            outerParent = outerParent->parent;
        }

        if (outerParent && !backendCtx.getLoopInfo(outerParent).exclusive.empty()) {
            rewriter.InsertTextBefore(stmt.getBeginLoc(), exclusiveNullText);
        }
    }

    // Bottom most `@inner` loop
    if (loopInfo->children.empty()) {
        // Get `@outer` attributed loop
        auto outerParent = parent;
        while (outerParent && !outerParent->hasOuter()) {
            outerParent = outerParent->parent;
        }

        if (outerParent && !backendCtx.getLoopInfo(outerParent).exclusive.empty()) {
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