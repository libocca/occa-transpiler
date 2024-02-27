#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/range_to_string.h"

namespace {
using namespace oklt;
using namespace clang;

std::string buildBinOp(const BinaryOperator* binOp, const Attr* attr, SessionStage& stage) {
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
               right)
        .value();
}

std::string buildUnOp(const UnaryOperator* unOp, const Attr* attr, SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto expr = getSourceText(*unOp->getSubExpr(), ctx);
    auto op = unOp->getOpcodeStr(unOp->getOpcode()).str();
    auto typeStr = unOp->getType().getAsString();
    return util::fmt(
               "sycl::atomic_ref<{},sycl::memory_order::relaxed,sycl::memory_scope::device,sycl::"
               "access::address_space::global_space>({}){}",
               typeStr,
               expr,
               op)
        .value();
}

HandleResult handleAtomicAttribute(const clang::Attr* attr,
                                   const clang::Stmt* stmt,
                                   SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto newExpression = [&]() -> tl::expected<std::string, Error> {
        if (isa<BinaryOperator>(stmt)) {
            const BinaryOperator* binOp = cast<BinaryOperator>(stmt);
            return buildBinOp(binOp, attr, stage);
        }
        if (isa<UnaryOperator>(stmt)) {
            const UnaryOperator* unOp = cast<UnaryOperator>(stmt);
            return buildUnOp(unOp, attr, stage);
        }

        return getSourceText(*dyn_cast<Expr>(stmt), ctx);
    }();
    if (!newExpression) {
        return tl::make_unexpected(newExpression.error());
    }

    auto& rewriter = stage.getRewriter();
    SourceRange range(attr->getRange().getBegin().getLocWithOffset(-2), stmt->getEndLoc());
    rewriter.ReplaceText(range, newExpression.value());

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] DPCPP: Handle @atomic.\n";
#endif
    return true;
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
