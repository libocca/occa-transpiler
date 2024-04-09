#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"
#include "pipeline/core/error_codes.h"

#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>

namespace {
using namespace oklt;
using namespace clang;
using BinOpMapT = std::map<BinaryOperatorKind, std::string>;
using UnaryOpMapT = std::map<UnaryOperatorKind, std::string>;

HandleResult handleBinOp(SessionStage& stage,
                         const BinaryOperator& binOp,
                         const Attr& attr,
                         const BinOpMapT& binaryConvertMap) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto it = binaryConvertMap.find(binOp.getOpcode());
    if (it == binaryConvertMap.end()) {
        auto binOpStr = getSourceText(binOp, ctx);
        std::string description = "Atomic does not support this operation: " + binOpStr;
        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NOT_SUPPORTED_OP), description});
    }

    auto left = binOp.getLHS();
    if (!left->isLValue()) {
        auto leftText = getSourceText(*left, ctx);
        std::string description = leftText + ": is not lvalue";
        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NON_LVALUE_EXPR), description});
    }

    auto leftText = getSourceText(*left, ctx);
    auto right = binOp.getRHS();
    auto rigthText = getSourceText(*right, ctx);
    std::string atomicOpText = it->second + "(&(" + leftText + "), " + rigthText + ")";

    stage.getRewriter().ReplaceText({getAttrFullSourceRange(attr).getBegin(), binOp.getEndLoc()},
                                    atomicOpText);
    return {};
}

HandleResult handleUnOp(SessionStage& stage,
                        const UnaryOperator& unOp,
                        const Attr& attr,
                        const UnaryOpMapT& atomicUnaryMap) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto it = atomicUnaryMap.find(unOp.getOpcode());
    if (it == atomicUnaryMap.end()) {
        auto binOpStr = getSourceText(unOp, ctx);
        std::string description = "Atomic does not support this operation: " + binOpStr;
        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NOT_SUPPORTED_OP), description});
    }
    auto expr = unOp.getSubExpr();
    auto unOpText = getSourceText(*expr, ctx);
    std::string atomicOpText = it->second + "(&(" + unOpText + "), 1)";

    stage.getRewriter().ReplaceText({getAttrFullSourceRange(attr).getBegin(), unOp.getEndLoc()},
                                    atomicOpText);

    return {};
}

HandleResult handleCXXCopyOp(SessionStage& stage,
                             const CXXOperatorCallExpr& assignOp,
                             const Attr& attr,
                             const BinOpMapT& binaryConvertMap) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto numArgs = assignOp.getNumArgs();
    if (assignOp.getOperator() != OverloadedOperatorKind::OO_Equal || numArgs != 2) {
        auto exprStr = getSourceText(assignOp, ctx);
        std::string description = "Atomic does not support this operation: " + exprStr;
        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NOT_SUPPORTED_OP), description});
    }

    auto left = assignOp.getArg(0);
    auto right = assignOp.getArg(1);
    if (!left->isLValue()) {
        auto leftText = getSourceText(*left, ctx);
        std::string description = leftText + ": is not lvalue";
        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NON_LVALUE_EXPR), description});
    }

    auto leftText = getSourceText(*left, ctx);
    auto rigthText = getSourceText(*right, ctx);
    auto atomicFunc = binaryConvertMap.at(BinaryOperatorKind::BO_Assign);
    std::string atomicOpText = atomicFunc + "(&(" + leftText + "), " + rigthText + ")";

    stage.getRewriter().ReplaceText({getAttrFullSourceRange(attr).getBegin(), assignOp.getEndLoc()},
                                    atomicOpText);

    return {};
}

}  // namespace

namespace oklt::cuda_subset {

HandleResult handleAtomicAttribute(SessionStage& stage, const Stmt& stmt, const Attr& attr) {
    static const BinOpMapT atomicBinaryMap = {
        {BinaryOperatorKind::BO_Assign, "atomicExch"},
        {BinaryOperatorKind::BO_AddAssign, "atomicAdd"},
        {BinaryOperatorKind::BO_SubAssign, "atomicSub"},
        {BinaryOperatorKind::BO_AndAssign, "atomicAnd"},
        {BinaryOperatorKind::BO_OrAssign, "atomicOr"},
        {BinaryOperatorKind::BO_XorAssign, "atomicXor"},
    };
    static const UnaryOpMapT atomicUnaryMap = {
        {UnaryOperatorKind::UO_PreDec, "atomicDec"},
        {UnaryOperatorKind::UO_PostDec, "atomicDec"},
        {UnaryOperatorKind::UO_PreInc, "atomicInc"},
        {UnaryOperatorKind::UO_PostInc, "atomicInc"},
    };

    auto& ctx = stage.getCompiler().getASTContext();
    if (const auto binOp = dyn_cast_or_null<BinaryOperator>(&stmt)) {
        return handleBinOp(stage, *binOp, attr, atomicBinaryMap);
    }

    if (const auto unOp = dyn_cast_or_null<UnaryOperator>(&stmt)) {
        return handleUnOp(stage, *unOp, attr, atomicUnaryMap);
    }

    if (const auto assignOp = dyn_cast_or_null<CXXOperatorCallExpr>(&stmt)) {
        return handleCXXCopyOp(stage, *assignOp, attr, atomicBinaryMap);
    }

    // INFO: for Expr there must be used different overloaded getSourceText method
    if (const auto expr = dyn_cast_or_null<Expr>(&stmt)) {
        auto exprStr = getSourceText(*expr, ctx);
        std::string description = "Atomic does not support this operation: " + exprStr;

        return tl::make_unexpected(
            Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NOT_SUPPORTED_OP), description});
    }

    // INFO: looks like it's really statemet that actually should not happen
    auto stmtStr = getSourceText(stmt.getSourceRange(), ctx);
    std::string description = "Atomic does not support this operation: " + stmtStr;

    return tl::make_unexpected(
        Error{make_error_code(OkltPipelineErrorCode::ATOMIC_NOT_SUPPORTED_OP), description});
}
}  // namespace oklt::cuda_subset
