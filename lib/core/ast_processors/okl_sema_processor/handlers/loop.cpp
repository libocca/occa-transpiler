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

HandleResult preValidateOklForLoop(const Attr& attr,
                                   const ForStmt& stmt,
                                   OklSemaCtx& sema,
                                   SessionStage& stage) {
    SPDLOG_DEBUG("[Sema] Prevalidate loop with attribute {}", attr.getNormalizedFullName());
    auto params = stage.getAttrManager().parseAttr(attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto ok = sema.startParsingAttributedForLoop(&attr, stmt, &params.value(), stage);
    if (!ok) {
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult preValidateOklForLoopWithoutAttribute(const Attr* attr,
                                                   const ForStmt& stmt,
                                                   OklSemaCtx& sema,
                                                   SessionStage& stage) {
    // Process only not attributed for statements in kernel methods
    if (attr || !sema.isParsingOklKernel()) {
        return {};
    }

    SPDLOG_DEBUG("[Sema] Prevalidate loop without attribute");

    auto ok = sema.startParsingAttributedForLoop(nullptr, stmt, nullptr, stage);
    if (!ok) {
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult postValidateOklForLoop(const Attr& attr,
                                    const clang::ForStmt& stmt,
                                    OklSemaCtx& sema,
                                    SessionStage& stage) {
    SPDLOG_DEBUG("[Sema] postValidate loop with attribute {}", attr.getNormalizedFullName());
    auto params = stage.getAttrManager().parseAttr(attr, stage);
    if (!params) {
        return tl::make_unexpected(std::move(params.error()));
    }

    auto ok = sema.stopParsingAttributedForLoop(&attr, stmt, &params.value());
    if (!ok) {
        // make appropriate error code
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

HandleResult postValidateOklForLoopWithoutAttribute(const Attr* attr,
                                                    const clang::ForStmt& stmt,
                                                    OklSemaCtx& sema,
                                                    SessionStage& stage) {
    // Process only not attributed for statements in kernel methods
    if (attr || !sema.isParsingOklKernel()) {
        return {};
    }

    SPDLOG_DEBUG("[Sema] Postvalidate loop without attribute");

    auto ok = sema.stopParsingAttributedForLoop(nullptr, stmt, nullptr);
    if (!ok) {
        // make appropriate error code
        return tl::make_unexpected(std::move(ok.error()));
    }

    return ok;
}

}  // namespace oklt
