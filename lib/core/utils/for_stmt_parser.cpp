#include <oklt/core/error.h>
#include "attributes/frontend/params/tile.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/range_to_string.h"

#include <clang/AST/AST.h>
#include <clang/AST/ParentMapContext.h>

#include <tl/expected.hpp>

namespace {
using namespace oklt;
using namespace clang;

BinOp toOkl(BinaryOperatorKind bok) {
    static std::map<BinaryOperatorKind, BinOp> clang2okl = {
        {BinaryOperatorKind::BO_EQ, BinOp::Eq},
        {BinaryOperatorKind::BO_LE, BinOp::Le},
        {BinaryOperatorKind::BO_LT, BinOp::Lt},
        {BinaryOperatorKind::BO_GT, BinOp::Gt},
        {BinaryOperatorKind::BO_GE, BinOp::Ge},
        {BinaryOperatorKind::BO_AddAssign, BinOp::AddAssign},
        {BinaryOperatorKind::BO_RemAssign, BinOp::RemoveAssign},
    };
    auto it = clang2okl.find(bok);
    return it != clang2okl.end() ? it->second : BinOp::Other;
}

UnOp toOkl(UnaryOperatorKind uok) {
    static std::map<UnaryOperatorKind, UnOp> clang2okl = {
        {UnaryOperatorKind::UO_PreInc, UnOp::PreInc},
        {UnaryOperatorKind::UO_PostInc, UnOp::PostInc},
        {UnaryOperatorKind::UO_PreDec, UnOp::PreDec},
        {UnaryOperatorKind::UO_PostDec, UnOp::PostDec}};
    auto it = clang2okl.find(uok);
    return it != clang2okl.end() ? it->second : UnOp::Other;
}

bool EvaluateAsSizeT(const Expr* E, llvm::APSInt& Into, const ASTContext& ctx) {
    unsigned BitsInSizeT = ctx.getTypeSize(ctx.getSizeType());

    Expr::EvalResult ExprResult;
    if (!E->EvaluateAsInt(ExprResult, ctx, Expr::SE_AllowSideEffects)) {
        return false;
    }

    Into = ExprResult.Val.getInt();
    if (Into.isNegative() || !Into.isIntN(BitsInSizeT)) {
        return false;
    }

    Into = Into.zext(BitsInSizeT);
    return true;
};

}  // namespace

namespace oklt {
tl::expected<OklLoopInfo, Error> parseForStmt(const clang::Attr& a,
                                              const clang::ForStmt& s,
                                              SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    OklLoopInfo ret{.attr = a, .stmt = s};
    const Expr *start, *end = nullptr;

    if (isa<DeclStmt>(s.getInit())) {
        auto d = dyn_cast<DeclStmt>(s.getInit());
        if (!d->isSingleDecl()) {
            // TODO: add error code
            return tl::make_unexpected(Error{std::error_code(), "loop parse: multi-declaration"});
        }

        auto node = dyn_cast<VarDecl>(d->getSingleDecl());
        if (!node) {
            // TODO: add error code
            return tl::make_unexpected(Error{std::error_code(), "loop parse: no init"});
        }

        ret.var.name = node->getDeclName().getAsString();
        ret.var.varDecl = node;
        ret.var.typeName = node->getType().getAsString();

        start = node->getInit();
        while (auto rsh = dyn_cast_or_null<CastExpr>(start)) {
            start = rsh->getSubExpr();
        }
        ret.range.start = start;

        auto child_count = std::distance(start->children().begin(), start->children().end());
    }

    // Condition
    if (isa<BinaryOperator>(s.getCond())) {
        auto node = dyn_cast<BinaryOperator>(s.getCond());
        if (!node->isComparisonOp()) {
            // TODO: add error code
            return tl::make_unexpected(Error{std::error_code(), "loop parse: not comparison op"});
        }

        ret.condition.op = toOkl(node->getOpcode());
        ret.condition.cmp_ = node;

        // LSH
        auto lsh = dyn_cast_or_null<CastExpr>(node->getLHS());
        while (lsh && lsh->getSubExpr() && isa<CastExpr>(lsh->getSubExpr())) {
            lsh = dyn_cast_or_null<CastExpr>(lsh->getSubExpr());
        };
        if (lsh && lsh->getSubExpr()) {
            auto decl = dyn_cast_or_null<DeclRefExpr>(lsh->getSubExpr());
            if (decl && decl->getNameInfo().getAsString() == ret.var.name) {
                end = node->getRHS();
                ret.range.end = end;
            }
        };

        // RSH
        auto rsh = dyn_cast_or_null<CastExpr>(node->getRHS());
        while (rsh && rsh->getSubExpr() && isa<CastExpr>(rsh->getSubExpr())) {
            rsh = dyn_cast_or_null<CastExpr>(rsh->getSubExpr());
        };
        if (rsh && rsh->getSubExpr()) {
            auto decl = dyn_cast_or_null<DeclRefExpr>(rsh->getSubExpr());
            if (decl && decl->getNameInfo().getAsString() == ret.var.name) {
                end = node->getLHS();
                ret.range.end = end;
                ret.condition.op = toOkl(BinaryOperator::reverseComparisonOp(node->getOpcode()));
            }
        }

        if (!end) {
            // TODO: add error code
            return tl::make_unexpected(
                Error{std::error_code(), "loop parse: cond without init var"});
        }
    }

    // Increment
    if (isa<UnaryOperator>(s.getInc())) {
        auto node = dyn_cast<UnaryOperator>(s.getInc());
        ret.inc.op.uo = toOkl(node->getOpcode());
    } else if (isa<CompoundAssignOperator>(s.getInc())) {
        auto node = dyn_cast<CompoundAssignOperator>(s.getInc());

        auto lsh = dyn_cast_or_null<DeclRefExpr>(node->getLHS());
        if (lsh && lsh->getNameInfo().getAsString() != ret.var.name) {
            // TODO: add error code
            return tl::make_unexpected(Error{std::error_code(), "loop parse: decl not inc"});
        }

        ret.inc.op.bo = toOkl(node->getOpcode());
        ret.inc.val = node->getRHS();
    }

    ret.range.size = 0;

    // Determinate range size
    llvm::APSInt start_i, end_i;
    if (EvaluateAsSizeT(start, start_i, ctx) && EvaluateAsSizeT(end, end_i, ctx)) {
        start_i.setIsSigned(true);
        end_i.setIsSigned(true);
        if (ret.IsInc()) {
            end_i -= start_i;
            ret.range.size = end_i.getZExtValue();
        } else {
            start_i -= end_i;
            ret.range.size = start_i.getZExtValue();
        }
    }

    // Ugly way to retireve tile size
    auto& am = stage.getAttrManager();
    auto params = am.parseAttr(a, stage);
    if (params && params->type() == typeid(TileParams)) {
        ret.tileSize = std::any_cast<TileParams>(am.parseAttr(a, stage).value()).tileSize;
    }

    return ret;
}
}  // namespace oklt
