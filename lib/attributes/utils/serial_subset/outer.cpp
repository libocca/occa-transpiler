#include "attributes/frontend/params/loop.h"
#include "core/handler_manager/handler_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include <spdlog/spdlog.h>

namespace oklt::serial_subset {
using namespace clang;

const std::string exlusiveBeginText = "\nint _occa_exclusive_index;\n";

HandleResult handleOuterAttribute(SessionStage& s,
                                  const ForStmt& stmt,
                                  const Attr& a,
                                  const AttributedLoop* params) {
    SPDLOG_DEBUG("Handle [@outer] attribute");

    if (!params) {
        return tl::make_unexpected(Error{std::error_code(), "@outer params nullptr"});
    }

    auto& sema = s.tryEmplaceUserCtx<OklSemaCtx>();
    auto loopInfo = sema.getLoopInfo(stmt);
    if (!loopInfo) {
        return tl::make_unexpected(Error{{}, "@outer: failed to fetch loop meta data from sema"});
    }

    removeAttribute(s, a);

    auto compStmt = dyn_cast_or_null<CompoundStmt>(stmt.getBody());
    if (loopInfo->exclusiveInfo.declared) {
        auto indexLoc = compStmt->getLBracLoc().getLocWithOffset(1);
        s.getRewriter().InsertTextAfter(indexLoc, exlusiveBeginText);
    }

    return {};
}

}  // namespace oklt::serial_subset
