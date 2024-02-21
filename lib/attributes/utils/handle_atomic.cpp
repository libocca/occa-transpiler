#include "attributes/utils/handle_atomic.h"
#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"
#include "core/utils/range_to_string.h"
#include "pipeline/stages/transpiler/error_codes.h"

namespace oklt {

using namespace clang;

bool handleAtomicAttribute(const clang::Attr* attr, const clang::Stmt* stmt, SessionStage& stage) {
    using BinOpMapT = std::map<BinaryOperatorKind, std::string>;
    using UnaryOpMapT = std::map<UnaryOperatorKind, std::string>;
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
    if (isa<BinaryOperator>(stmt)) {
        const BinaryOperator* binOp = cast<BinaryOperator>(stmt);
        auto it = atomicBinaryMap.find(binOp->getOpcode());
        if (it == atomicBinaryMap.end()) {
            auto binOpStr = getSourceText(*binOp, ctx);
            std::string description = "Atomic does not support this operation: " + binOpStr;
            stage.pushError(make_error_code(OkltTranspilerErrorCode::ATOMIC_NOT_SUPPORTED_OP),
                            description);
            return false;
        }

        auto left = binOp->getLHS();
        if (!left->isLValue()) {
            auto leftText = getSourceText(*left, ctx);
            std::string description = leftText + ": is not lvalue";
            stage.pushError(make_error_code(OkltTranspilerErrorCode::ATOMIC_NON_LVALUE_EXPR),
                            description);
            return false;
        }

        auto& rewriter = stage.getRewriter();
        auto leftText = getSourceText(*left, ctx);
        auto right = binOp->getRHS();
        auto rigthText = getSourceText(*right, ctx);
        std::string atomicOpText = it->second + "(&(" + leftText + "), " + rigthText + ")";
        removeAttribute(attr, stage);
        rewriter.ReplaceText(stmt->getSourceRange(), atomicOpText);
        return true;
    }

    if (isa<UnaryOperator>(stmt)) {
        const UnaryOperator* unOp = cast<UnaryOperator>(stmt);
        auto it = atomicUnaryMap.find(unOp->getOpcode());
        if (it == atomicUnaryMap.end()) {
            auto binOpStr = getSourceText(*unOp, ctx);
            std::string description = "Atomic does not support this operation: " + binOpStr;
            stage.pushError(make_error_code(OkltTranspilerErrorCode::ATOMIC_NOT_SUPPORTED_OP),
                            description);
            return false;
        }
        auto expr = unOp->getSubExpr();
        auto unOpText = getSourceText(*expr, ctx);
        auto& rewriter = stage.getRewriter();
        std::string atomicOpText = it->second + "(&(" + unOpText + "), 1)";
        removeAttribute(attr, stage);
        rewriter.ReplaceText(stmt->getSourceRange(), atomicOpText);
        return true;
    }

    if (isa<CXXOperatorCallExpr>(stmt)) {
        auto assignOp = cast<CXXOperatorCallExpr>(stmt);
        auto numArgs = assignOp->getNumArgs();
        if (assignOp->getOperator() == OverloadedOperatorKind::OO_Equal && numArgs == 2) {
            auto left = assignOp->getArg(0);
            auto right = assignOp->getArg(1);
            if (!left->isLValue()) {
                auto leftText = getSourceText(*left, ctx);
                std::string description = leftText + ": is not lvalue";
                stage.pushError(make_error_code(OkltTranspilerErrorCode::ATOMIC_NON_LVALUE_EXPR),
                                description);
                return false;
            }
            auto& rewriter = stage.getRewriter();
            auto leftText = getSourceText(*left, ctx);
            auto rigthText = getSourceText(*right, ctx);
            auto atomicFunc = atomicBinaryMap.at(BinaryOperatorKind::BO_Assign);
            std::string atomicOpText = atomicFunc + "(&(" + leftText + "), " + rigthText + ")";
            removeAttribute(attr, stage);
            rewriter.ReplaceText(stmt->getSourceRange(), atomicOpText);
            return true;
        }
    }

    // INFO: for Expr there must be used different overloaded getSourceText method
    if (isa<Expr>(stmt)) {
        const Expr* expr = cast<Expr>(stmt);
        auto exprStr = getSourceText(*expr, ctx);
        std::string description = "Atomic does not support this operation: " + exprStr;
        stage.pushError(make_error_code(OkltTranspilerErrorCode::ATOMIC_NOT_SUPPORTED_OP),
                        description);
        return false;
    }

    // INFO: looks like it's really statemet that actually should not happen
    auto binOpStr = getSourceText(stmt->getSourceRange(), ctx);
    std::string description = "Atomic does not support this operation: " + binOpStr;
    stage.pushError(make_error_code(OkltTranspilerErrorCode::ATOMIC_NOT_SUPPORTED_OP), description);
    return false;
}
}  // namespace oklt
