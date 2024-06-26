#include "attributes/frontend/params/loop.h"
#include "core/handler_manager/handler_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

namespace {
const std::string exclusiveNullText = "_occa_exclusive_index = 0;\n";
const std::string exclusiveIncText = "++_occa_exclusive_index;\n";
}  // namespace

HandleResult handleInnerAttribute(SessionStage& s,
                                  const ForStmt& stmt,
                                  const Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@inner] attribute");

    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@inner params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@inner: failed to fetch loop meta data from sema"});
    }

    auto& rewriter = s.getRewriter();

    removeAttribute(s, a);

    // Top most `@inner` loop
    if (auto parent = loopInfo->getAttributedParent(); parent->has(LoopType::Outer)) {
        if (parent->exclusiveInfo.declared) {
            rewriter.InsertTextBefore(stmt.getBeginLoc(), exclusiveNullText);
        }
    }

    // Bottom most `@inner` loop
    if (loopInfo->children.empty()) {
        // Get `@outer` attributed loop
        auto parent =
            loopInfo->getAttributedParent([](OklLoopInfo& v) { return v.has(LoopType::Outer); });
        if (parent && parent->exclusiveInfo.declared) {
            auto compStmt = dyn_cast_or_null<CompoundStmt>(loopInfo->stmt.getBody());
            SourceLocation incLoc =
                compStmt ? compStmt->getRBracLoc().getLocWithOffset(-1) : stmt.getEndLoc();
            rewriter.InsertTextBefore(incLoc, exclusiveIncText);
        }
    }

    return {};
}

}  // namespace oklt::serial_subset
