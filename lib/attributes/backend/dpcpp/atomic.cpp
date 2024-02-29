#include <core/transpilation.h>
#include <core/transpilation_encoded_names.h>
#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/range_to_string.h"
#include "pipeline/stages/transpiler/error_codes.h"

namespace {
using namespace oklt;
using namespace clang;

tl::expected<std::string, Error> buildBinOp(const BinaryOperator* binOp,
                                            const Attr& attr,
                                            SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto left = getSourceText(*binOp->getLHS(), ctx);
    auto right = getSourceText(*binOp->getRHS(), ctx);
    auto op = binOp->getOpcodeStr().str();
    auto typeStr = binOp->getType().getAsString();
    return util::fmt(
        "sycl::atomic_ref<{},sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::"
        "access::address_space::global_space>({}) {} {}",
        typeStr,
        left,
        op,
        right);
}

tl::expected<std::string, Error> buildUnOp(const UnaryOperator* unOp,
                                           const Attr& attr,
                                           SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto expr = getSourceText(*unOp->getSubExpr(), ctx);
    auto op = unOp->getOpcodeStr(unOp->getOpcode()).str();
    auto typeStr = unOp->getType().getAsString();
    return util::fmt(
        "sycl::atomic_ref<{},sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::"
        "access::address_space::global_space>({}){}",
        typeStr,
        expr,
        op);
}

tl::expected<std::string, Error> buildCXXCopyOp(const CXXOperatorCallExpr* op,
                                                const Attr& attr,
                                                SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto numArgs = op->getNumArgs();
    if (op->getOperator() != OverloadedOperatorKind::OO_Equal || numArgs != 2) {
        auto exprStr = getSourceText(*op, ctx);
        return tl::make_unexpected(
            Error{make_error_code(OkltTranspilerErrorCode::ATOMIC_NOT_SUPPORTED_OP),
                  "Unsupported atomic operation: " + exprStr});
    }
    auto left = op->getArg(0);
    auto right = op->getArg(1);
    auto leftStr = getSourceText(*left, ctx);
    auto rigthStr = getSourceText(*right, ctx);
    auto typeStr = op->getType().getAsString();

    if (!left->isLValue()) {
        auto leftText = getSourceText(*left, ctx);
        return tl::make_unexpected(
            Error{make_error_code(OkltTranspilerErrorCode::ATOMIC_NON_LVALUE_EXPR),
                  leftText + ": is not lvalue"});
    }
    return util::fmt(
        "sycl::atomic_ref<{},sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::"
        "access::address_space::global_space>({}) = {}",
        typeStr,
        leftStr,
        rigthStr);
}

HandleResult handleAtomicAttribute(const clang::Attr& attr,
                                   const clang::Stmt& stmt,
                                   SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto newExpression = [&]() -> tl::expected<std::string, Error> {
        if (isa<BinaryOperator>(stmt)) {
            const BinaryOperator* binOp = cast<BinaryOperator>(&stmt);
            return buildBinOp(binOp, attr, stage);
        }
        if (isa<UnaryOperator>(stmt)) {
            const UnaryOperator* unOp = cast<UnaryOperator>(&stmt);
            return buildUnOp(unOp, attr, stage);
        }
        if (isa<CXXOperatorCallExpr>(stmt)) {
            const CXXOperatorCallExpr* op = cast<CXXOperatorCallExpr>(&stmt);
            return buildCXXCopyOp(op, attr, stage);
        }

        return getSourceText(*dyn_cast<Expr>(&stmt), ctx);
    }();
    if (!newExpression) {
        return tl::make_unexpected(newExpression.error());
    }

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle @atomic.\n";
#endif

    SourceRange range(attr.getRange().getBegin().getLocWithOffset(-2), stmt.getEndLoc());
    return TranspilationBuilder(
               stage.getCompiler().getSourceManager(), attr.getNormalizedFullName(), 1)
        .addReplacement(OKL_ATOMIC_OP, range, newExpression.value())
        .build();
}

__attribute__((constructor)) void registerAttrBackend() {
    auto ok = oklt::AttributeManager::instance().registerBackendHandler(
        {TargetBackend::DPCPP, ATOMIC_ATTR_NAME}, makeSpecificAttrHandle(handleAtomicAttribute));

    if (!ok) {
        llvm::errs() << "failed to register " << ATOMIC_ATTR_NAME
                     << " attribute handler for DPCPP backend\n";
    }
}
}  // namespace
