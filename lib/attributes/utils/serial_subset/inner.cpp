#include "attributes/utils/serial_subset/common.h"

namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string exclusiveNullText = "_occa_exclusive_index = 0;\n";
const std::string exclusiveIncText = "++_occa_exclusive_index;\n";
}  // namespace

HandleResult handleInnerAttribute(const Attr& a,
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

    auto& backendCtx = getBackendCtxFromStage(s);
    auto& rewriter = s.getRewriter();

    removeAttribute(a, s);

    // Top most `@inner` loop
    if (auto parent = loopInfo->getAttributedParent(); parent->has(LoopType::Outer)) {
        if (!backendCtx.getLoopInfo(parent).exclusive.empty()) {
            rewriter.InsertTextBefore(stmt.getBeginLoc(), exclusiveNullText);
        }
    }

    // Bottom most `@inner` loop
    if (loopInfo->children.empty()) {
        // Get `@outer` attributed loop
        auto parent = loopInfo->getAttributedParent([](OklLoopInfo& v) { return v.has(LoopType::Outer); });
        if (parent && !backendCtx.getLoopInfo(parent).exclusive.empty()) {
            auto compStmt = dyn_cast_or_null<CompoundStmt>(loopInfo->stmt.getBody());
            SourceLocation incLoc =
                compStmt ? compStmt->getRBracLoc().getLocWithOffset(-1) : stmt.getEndLoc();
            rewriter.InsertTextBefore(incLoc, exclusiveIncText);
        }
    }

    return {};
}

}  // namespace oklt::serial_subset
