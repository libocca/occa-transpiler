#include "attributes/attribute_names.h"
#include "core/handler_manager/backend_handler.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"
#include "pipeline/core/error_codes.h"

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

tl::expected<std::string, Error> buildBinOp(SessionStage& stage,
                                            const Attr& attr,
                                            const BinaryOperator& binOp) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto left = getSourceText(*binOp.getLHS(), ctx);
    auto right = getSourceText(*binOp.getRHS(), ctx);
    auto op = binOp.getOpcodeStr().str();
    auto typeStr = binOp.getType().getAsString();
    return util::fmt(
        "sycl::atomic_ref<{},sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::"
        "access::address_space::global_space>({}) {} {}",
        typeStr,
        left,
        op,
        right);
}

tl::expected<std::string, Error> buildUnOp(SessionStage& stage,
                                           const Attr& attr,
                                           const UnaryOperator& unOp) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto expr = getSourceText(*unOp.getSubExpr(), ctx);
    auto op = unOp.getOpcodeStr(unOp.getOpcode()).str();
    auto typeStr = unOp.getType().getAsString();
    return util::fmt(
        "sycl::atomic_ref<{},sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::"
        "access::address_space::global_space>({}){}",
        typeStr,
        expr,
        op);
}

tl::expected<std::string, Error> buildCXXCopyOp(SessionStage& stage,
                                                const Attr& attr,
                                                const CXXOperatorCallExpr& op) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto numArgs = op.getNumArgs();
    if (op.getOperator() != OverloadedOperatorKind::OO_Equal || numArgs != 2) {
        auto exprStr = getSourceText(op, ctx);
        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NOT_SUPPORTED_OP),
                  "Unsupported atomic operation: " + exprStr});
    }
    auto left = op.getArg(0);
    auto right = op.getArg(1);
    auto leftStr = getSourceText(*left, ctx);
    auto rigthStr = getSourceText(*right, ctx);
    auto typeStr = op.getType().getAsString();

    if (!left->isLValue()) {
        auto leftText = getSourceText(*left, ctx);
        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NON_LVALUE_EXPR),
                  leftText + ": is not lvalue"});
    }
    return util::fmt(
        "sycl::atomic_ref<{},sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::"
        "access::address_space::global_space>({}) = {}",
        typeStr,
        leftStr,
        rigthStr);
}

tl::expected<std::string, Error> buildNewExpression(SessionStage& stage,
                                                    const Attr& attr,
                                                    const clang::Stmt& stmt);

// TODO: Replicates legacy occa transpiler behavior. Seems to be incorrect that each statement is
// made atomic, and not the whole block.
tl::expected<std::string, Error> buildCompoundStmt(SessionStage& stage,
                                                   const Attr& attr,
                                                   const CompoundStmt& op) {
    std::string res = "{";
    for (auto& stmt : op.body()) {
        auto newExpr = buildNewExpression(stage, attr, *stmt);
        if (!newExpr) {
            return tl::make_unexpected(newExpr.error());
        }
        res += newExpr.value() + ";\n";
    }
    res += "}";
    return res;
}

tl::expected<std::string, Error> buildNewExpression(SessionStage& stage,
                                                    const Attr& attr,
                                                    const clang::Stmt& stmt) {
    if (isa<BinaryOperator>(stmt)) {
        const BinaryOperator* binOp = cast<BinaryOperator>(&stmt);
        return buildBinOp(stage, attr, *binOp);
    }
    if (isa<UnaryOperator>(stmt)) {
        const UnaryOperator* unOp = cast<UnaryOperator>(&stmt);
        return buildUnOp(stage, attr, *unOp);
    }
    if (isa<CXXOperatorCallExpr>(stmt)) {
        const CXXOperatorCallExpr* op = cast<CXXOperatorCallExpr>(&stmt);
        return buildCXXCopyOp(stage, attr, *op);
    }
    if (isa<CompoundStmt>(stmt)) {
        const CompoundStmt* compoundStmt = cast<CompoundStmt>(&stmt);
        return buildCompoundStmt(stage, attr, *compoundStmt);
    }
    return tl::make_unexpected(Error{{}, "Error: Unable to transform general @atomic code"});
}

HandleResult handleAtomicAttribute(SessionStage& stage,
                                   const clang::Stmt& stmt,
                                   const clang::Attr& attr) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto newExpression = buildNewExpression(stage, attr, stmt);
    if (!newExpression) {
        return tl::make_unexpected(newExpression.error());
    }

    SPDLOG_DEBUG("Handle [@atomic] attribute");

    SourceRange range(getAttrFullSourceRange(attr).getBegin(), stmt.getEndLoc());
    stage.getRewriter().ReplaceText(range, newExpression.value());

    return {};
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = registerBackendHandler(TargetBackend::DPCPP, ATOMIC_ATTR_NAME, handleAtomicAttribute);

    if (!ok) {
        SPDLOG_ERROR("[DPCPP] Failed to register {} attribute handler", ATOMIC_ATTR_NAME);
    }
}
}  // namespace
