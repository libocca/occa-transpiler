#include "loop.h"
#include "core/handler_manager/handler_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>

#include "spdlog/spdlog.h"

namespace oklt {
using namespace clang;

HandleResult preValidateOklForLoop(SessionStage& stage,
                                   const ForStmt& stmt,
                                   const Attr& attr) {
    SPDLOG_DEBUG("[Sema] Pre validate loop with attribute {}", attr.getNormalizedFullName());
    auto params = stage.getAttrManager().parseAttr(stage, attr);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto ok = sema.startParsingAttributedForLoop(stage, stmt, &attr, &params.value());
    if (!ok) {
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult preValidateOklForLoopWithoutAttribute(SessionStage& stage,
                                                   const ForStmt& stmt,
                                                   const Attr* attr) {
    // Process only not attributed for statements in kernel methods
    if (attr) {
        return {};
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        return {};
    }

    SPDLOG_DEBUG("[Sema] Pre validate loop without attribute");

    auto ok = sema.startParsingAttributedForLoop(stage, stmt, nullptr, nullptr);
    if (!ok) {
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult postValidateOklForLoop(SessionStage& stage,
                                    const clang::ForStmt& stmt,
                                    const Attr& attr) {
    SPDLOG_DEBUG("[Sema] Post validate loop with attribute {}", attr.getNormalizedFullName());
    auto params = stage.getAttrManager().parseAttr(stage, attr);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    auto ok = sema.stopParsingAttributedForLoop(stmt, &attr, &params.value());
    if (!ok) {
        // make appropriate error code
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult postValidateOklForLoopWithoutAttribute(SessionStage& stage,
                                                    const clang::ForStmt& stmt,
                                                    const Attr* attr) {
    // Process only not attributed for statements in kernel methods
    if (attr) {
        return {};
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        return {};
    }

    SPDLOG_DEBUG("[Sema] Post validate loop without attribute");

    auto ok = sema.stopParsingAttributedForLoop(stmt, nullptr, nullptr);
    if (!ok) {
        // make appropriate error code
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

}  // namespace oklt
