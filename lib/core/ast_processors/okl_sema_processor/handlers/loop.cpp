#include <oklt/core/kernel_metadata.h>

#include "core/ast_processors/okl_sema_processor/handlers/loop.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>
#include <spdlog/spdlog.h>

namespace oklt {
using namespace clang;

HandleResult preValidateOklForLoop(SessionStage& stage,
                                   OklSemaCtx& sema,
                                   const ForStmt& stmt,
                                   const Attr& attr) {
    SPDLOG_DEBUG("[Sema] Prevalidate loop with attribute {}", attr.getNormalizedFullName());
    auto params = stage.getAttrManager().parseAttr(stage, attr);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto ok = sema.startParsingAttributedForLoop(stage, stmt, &attr, &params.value());
    if (!ok) {
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult preValidateOklForLoopWithoutAttribute(SessionStage& stage,
                                                   OklSemaCtx& sema,
                                                   const ForStmt& stmt,
                                                   const Attr* attr) {
    // Process only not attributed for statements in kernel methods
    if (attr || !sema.isParsingOklKernel()) {
        return {};
    }

    SPDLOG_DEBUG("[Sema] Prevalidate loop without attribute");

    auto ok = sema.startParsingAttributedForLoop(stage, stmt, nullptr, nullptr);
    if (!ok) {
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult postValidateOklForLoop(SessionStage& stage,
                                    OklSemaCtx& sema,
                                    const clang::ForStmt& stmt,
                                    const Attr& attr) {
    SPDLOG_DEBUG("[Sema] postValidate loop with attribute {}", attr.getNormalizedFullName());
    auto params = stage.getAttrManager().parseAttr(stage, attr);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto ok = sema.stopParsingAttributedForLoop(stmt, &attr, &params.value());
    if (!ok) {
        // make appropriate error code
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult postValidateOklForLoopWithoutAttribute(SessionStage& stage,
                                                    OklSemaCtx& sema,
                                                    const clang::ForStmt& stmt,
                                                    const Attr* attr) {
    // Process only not attributed for statements in kernel methods
    if (attr || !sema.isParsingOklKernel()) {
        return {};
    }

    SPDLOG_DEBUG("[Sema] Postvalidate loop without attribute");

    auto ok = sema.stopParsingAttributedForLoop(stmt, nullptr, nullptr);
    if (!ok) {
        // make appropriate error code
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

}  // namespace oklt
