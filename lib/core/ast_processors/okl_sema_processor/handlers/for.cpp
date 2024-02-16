#include <oklt/core/ast_processors/okl_sema_processor/okl_sema_ctx.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/attribute_names.h>
#include <oklt/core/metadata/program.h>
#include <oklt/core/transpiler_session/session_stage.h>

#include <clang/AST/AST.h>

#define OKL_SEMA_DEBUG

namespace {
using namespace oklt;
using namespace clang;

struct OklForStmt {
    const ForStmt* forStmt;
    const Attr* attr;
};

std::optional<OklForStmt> getOklForStmt(const AttributedStmt* attrStmt) {
    auto attrs = attrStmt->getAttrs();
    if (attrs.size() > 1) {
        return std::nullopt;
    }

    const auto& attrName = attrs[0]->getNormalizedFullName();
    if (attrName == TILE_ATTR_NAME || attrName == INNER_ATTR_NAME || attrName == OUTER_ATTR_NAME) {
        if (attrStmt->getSubStmt()->getStmtClass() != Stmt::ForStmtClass) {
            return std::nullopt;
        }
    }

    return OklForStmt{.forStmt = cast<ForStmt>(attrStmt->getSubStmt()), .attr = attrs[0]};
}

std::string prettyPrint(Stmt* S, const PrintingPolicy& policy) {
    std::string ret;
    if (!S) {
        return ret;
    }

    llvm::raw_string_ostream os(ret);
    S->printPretty(os, nullptr, policy);

    return ret;
};

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

LoopMetadata parseForStmt(const ForStmt* S, ASTContext& ctx) {
    LoopMetadata ret;
    Expr *start, *end;

    auto policy = ctx.getPrintingPolicy();

    if (isa<DeclStmt>(S->getInit())) {
        auto d = dyn_cast<DeclStmt>(S->getInit());
        if (!d->isSingleDecl()) {
            // TODO: throw Multi-Declaration
            return ret;
        }

        auto node = dyn_cast<VarDecl>(d->getSingleDecl());
        if (!node) {
            // TODO: throw No Init-statement
            return ret;
        }

        ret.name = node->getDeclName().getAsString();
        ret.type = node->getType().getAsString();

        start = node->getInit();
        while (auto rsh = dyn_cast_or_null<CastExpr>(start)) {
            start = rsh->getSubExpr();
        }
        ret.range.start = prettyPrint(start, policy);

        auto child_count = std::distance(start->children().begin(), start->children().end());
        if (child_count > 0 && !node->getInit()->isEvaluatable(ctx)) {
            ret.range.start = "(" + ret.range.start + ")";
        }
    }

    // Condition
    if (isa<BinaryOperator>(S->getCond())) {
        auto node = dyn_cast<BinaryOperator>(S->getCond());
        if (!node->isComparisonOp()) {
            // TODO: throw Non Comparison OP
            return ret;
        }

        ret.condition.op = node->getOpcode();
        ret.condition.cmp = prettyPrint(node, policy);

        // LSH
        auto lsh = dyn_cast_or_null<CastExpr>(node->getLHS());
        while (lsh && lsh->getSubExpr() && isa<CastExpr>(lsh->getSubExpr())) {
            lsh = dyn_cast_or_null<CastExpr>(lsh->getSubExpr());
        };
        if (lsh && lsh->getSubExpr()) {
            auto decl = dyn_cast_or_null<DeclRefExpr>(lsh->getSubExpr());
            if (decl && decl->getNameInfo().getAsString() == ret.name) {
                end = node->getRHS();
                ret.range.end = prettyPrint(end, policy);
            }
        };

        // RSH
        auto rsh = dyn_cast_or_null<CastExpr>(node->getRHS());
        while (rsh && rsh->getSubExpr() && isa<CastExpr>(rsh->getSubExpr())) {
            rsh = dyn_cast_or_null<CastExpr>(rsh->getSubExpr());
        };
        if (rsh && rsh->getSubExpr()) {
            auto decl = dyn_cast_or_null<DeclRefExpr>(rsh->getSubExpr());
            if (decl && decl->getNameInfo().getAsString() == ret.name) {
                end = node->getLHS();
                ret.range.end = prettyPrint(end, policy);
                ret.condition.op = BinaryOperator::reverseComparisonOp(node->getOpcode());
            }
        }

        if (!end) {
            // TODO: throw Condition not using init variable
            return ret;
        }
    }

    // Increment
    if (isa<UnaryOperator>(S->getInc())) {
        auto node = dyn_cast<UnaryOperator>(S->getInc());
        ret.inc.op.uo = node->getOpcode();
    } else if (isa<CompoundAssignOperator>(S->getInc())) {
        auto node = dyn_cast<CompoundAssignOperator>(S->getInc());

        auto lsh = dyn_cast_or_null<DeclRefExpr>(node->getLHS());
        if (lsh && lsh->getNameInfo().getAsString() != ret.name) {
            // TODO: throw Declaration is not incremented?
            return ret;
        }

        ret.inc.op.bo = node->getOpcode();
        ret.inc.val = prettyPrint(node->getRHS(), policy);
    }

    ret.range.size = 0;

    // Determinate range size
    llvm::APSInt start_i, end_i;
    if (EvaluateAsSizeT(start, start_i, ctx) && EvaluateAsSizeT(end, end_i, ctx)) {
        if (ret.IsInc()) {
            end_i -= start_i;
            ret.range.size = end_i.getZExtValue();
        } else {
            start_i -= end_i;
            ret.range.size = start_i.getZExtValue();
        }
    }

    return ret;
}

}  // namespace

namespace oklt {
bool prepareOklForStmt(const AttributedStmt* attrStmt, SessionStage& stage) {
    // expect single attribute
    auto oklForStmt = getOklForStmt(attrStmt);
    if (!oklForStmt) {
        return true;
    }

    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();
    if (!sema.isParsingOklKernel()) {
        //  make approptiate error code
        stage.pushError(std::error_code(), "okl loop attribute inside of non kernel function");
        return false;
    }

    auto result = sema.validateForSeries(oklForStmt->forStmt, oklForStmt->attr);
    if (!result) {
        //  make approptiate error code
        stage.pushError(result.error());
        return false;
    }

    return true;
}

bool transpileOklForStmt(const clang::AttributedStmt* attrStmt, SessionStage& stage) {
    auto oklForStmt = getOklForStmt(attrStmt);
    auto& sema = stage.tryEmplaceUserCtx<OklSemaCtx>();

    auto ret =
        AttributeManager::instance().handleAttr(oklForStmt->attr, oklForStmt->forStmt, stage);
    if (!ret) {
        return false;
    }

    auto loopMetaData = parseForStmt(oklForStmt->forStmt, stage.getCompiler().getASTContext());
    //TODO save in sema

    return true;
}
}  // namespace oklt
