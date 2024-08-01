#include <oklt/core/kernel_metadata.h>

#include "attributes/utils/default_handlers.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string outerLoopText = "\nint _occa_exclusive_index;";
const std::string exlusiveExprText = "[_occa_exclusive_index]";
}  // namespace

HandleResult handleExclusiveDeclAttribute(SessionStage& s, const Decl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute (Decl)");

    removeAttribute(s, a);
    return {};
}

HandleResult handleExclusiveVarAttribute(SessionStage& s, const VarDecl& decl, const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute (VarDecl)");

    removeAttribute(s, a);

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (loopInfo && loopInfo->isRegular()) {
        loopInfo = loopInfo->getAttributedParent();
    }
    if (loopInfo && loopInfo->has(LoopType::Inner)) {
        return tl::make_unexpected(
            Error{{}, "Cannot define [@exclusive] variables inside an [@inner] loop"});
    }
    auto child = loopInfo ? loopInfo->getFirstAttributedChild() : nullptr;
    bool isInnerChild = child && child->has(LoopType::Inner);
    if (!loopInfo || !loopInfo->has(LoopType::Outer) || !isInnerChild) {
        return tl::make_unexpected(
            Error{{}, "Must define [@exclusive] variables between [@outer] and [@inner] loops"});
    }

    auto& rewriter = s.getRewriter();

    // Find max size of inner loops
    size_t sz = 0;
    for (auto child : loopInfo->children) {
        auto v = child.getInnerSizes();
        if (v.hasNullOpts()) {
            sz = 1024;
            break;
        }
        sz = std::max(v.product(), sz);
    }
    std::string varSuffix = "[" + std::to_string(sz) + "]";

    // Add size and wrap initialization.
    auto nameLoc = decl.getLocation().getLocWithOffset(decl.getName().size());
    rewriter.InsertTextAfter(nameLoc, varSuffix);
    if (decl.hasInit()) {
        auto expr = decl.getInit();
        rewriter.InsertTextBefore(expr->getBeginLoc(), "{");
        rewriter.InsertTextAfter(decl.getEndLoc().getLocWithOffset(1), "}");
    }

    return defaultHandleExclusiveDeclAttribute(s, decl, a);
}

HandleResult handleExclusiveExprAttribute(SessionStage& s, const DeclRefExpr& expr, const Attr& a) {
    SPDLOG_DEBUG("Handle [@exclusive] attribute (DeclRefExpr)");

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo();
    if (!loopInfo) {
        return tl::make_unexpected(
            Error{{}, "@exclusive: failed to fetch loop meta data from sema"});
    }

    auto loc = expr.getLocation().getLocWithOffset(expr.getNameInfo().getAsString().size());
    s.getRewriter().InsertTextAfter(loc, exlusiveExprText);
    return defaultHandleExclusiveStmtAttribute(s, expr, a);
}

}  // namespace oklt::serial_subset
