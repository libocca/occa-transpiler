// #include <clang/AST/Decl.h>
// #include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>
// #include <oklt/attributes/frontend/parsers/tile.h>
// #include <oklt/core/attribute_manager/attribute_manager.h>
// #include <oklt/core/transpiler_session/session_stage.h>
#include "attributes/frontend/params/tile.h"
#include <oklt/util/string_utils.h>
#include <functional>
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "handle.h"

#include <clang/AST/Decl.h>
namespace oklt::cuda_subset {
using namespace clang;
namespace {

struct LoopMetadata {
    std::string type;
    std::string name;
    struct {
        std::string start;
        std::string end;
        size_t size = 0;
    } range;
    struct {
        std::string cmp;
        BinaryOperator::Opcode op = BO_EQ;
    } condition;
    struct {
        std::string val;
        union {
            UnaryOperator::Opcode uo;
            BinaryOperator::Opcode bo;
        } op;
    } inc;

    bool IsInc() const {
        bool ret = false;
        if (inc.val.empty()) {
            ret = (inc.op.uo == UO_PreInc || inc.op.uo == UO_PostInc);
        } else {
            ret = (inc.op.bo == BO_AddAssign);
        }
        ret = (ret && (condition.op == BO_LE || condition.op == BO_LT));

        return ret;
    };
    std::string getRangeSizeStr() const {
        if (IsInc()) {
            return range.end + " - " + range.start;
        } else {
            return range.start + " - " + range.end;
        };
    };

    bool isUnary() const {
        if (!inc.val.empty()) {
            return false;
        }
        // should by unnecessary check, but just in case
        return (inc.op.uo == UO_PreInc) || (inc.op.uo == UO_PostInc) || (inc.op.uo == UO_PreDec) ||
               (inc.op.uo == UO_PostDec);
    }
};

std::string prettyPrint(const Stmt* S, const PrintingPolicy& policy) {
    std::string ret = "";
    if (!S) {
        return ret;
    }

    llvm::raw_string_ostream os(ret);
    S->printPretty(os, nullptr, policy);

    return ret;
};

bool EvaluateAsSizeT(const Expr* E, llvm::APSInt& Into, ASTContext& ctx) {
    unsigned BitsInSizeT = ctx.getTypeSize(ctx.getSizeType());

    Expr::EvalResult ExprResult;
    if (!E->EvaluateAsInt(ExprResult, ctx, Expr::SE_AllowSideEffects))
        return false;
    Into = ExprResult.Val.getInt();
    if (Into.isNegative() || !Into.isIntN(BitsInSizeT))
        return false;
    Into = Into.zext(BitsInSizeT);
    return true;
};

LoopMetadata ParseForStmt(ForStmt* S, ASTContext& ctx) {
    auto& policy = ctx.getPrintingPolicy();

    LoopMetadata ret;
    Expr *start, *end;

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

std::string dimToStr(const Dim& dim) {
    static std::map<Dim, std::string> mapping{{Dim::X, "x"}, {Dim::Y, "y"}, {Dim::Z, "z"}};
    return mapping[dim];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strDim = dimToStr(loop.dim);
    switch (loop.type) {
        case (LoopType::Inner):
            return util::fmt("threadIdx.{}", dimToStr(loop.dim)).value();
        case (LoopType::Outer):
            return util::fmt("blockIdx.{}", dimToStr(loop.dim)).value();
        default:  // Incorrect case
            return "";
    }
}

std::string getTiledVariableName(const LoopMetadata& forLoop) {
    return "_occa_tiled_" + forLoop.name;
}

// Produces something like: int _occa_tiled_i = init + ((tileSize * inc) * threadIdx.x);
//                      or: int _occa_tiled_i = init + ((tileSize * inc) * blockIdx.x);
std::string innerOuterLoopIdxLineFirst(const LoopMetadata& forLoop,
                                       const AttributedLoop& loop,
                                       const TileParams* params,
                                       int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = {} + ({} * {});",
                                  forLoop.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = {} + (({} * {}) * {});",
                                  forLoop.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  params->tileSize,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    return res;
}

// Produces something like: int i = _occa_tiled_i + (inc * threadIdx.x);
std::string innerOuterLoopIdxLineSecond(const LoopMetadata& forLoop,
                                        const AttributedLoop& loop,
                                        const TileParams* params,
                                        int& openedScopeCounter) {
    static_cast<void>(params);
    auto tiledVar = getTiledVariableName(forLoop);

    std::string idx = getIdxVariable(loop);

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(
            util::fmt("{} {} = {} + {};", forLoop.type, forLoop.name, tiledVar, idx).value());
    } else {
        res = std::move(util::fmt("{} {} = {} + (({}) * {});",
                                  forLoop.type,
                                  forLoop.name,
                                  tiledVar,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return "{" + res;  // Open new scope
}

// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); ++i) {
std::string regularLoopIdxLineFirst(const LoopMetadata& forLoop,
                                    const AttributedLoop& regularLoop,
                                    const TileParams* params,
                                    int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto blockSize = std::to_string(params->tileSize);

    auto res = util::fmt("for({} {} = {}; {} < {}; {} += {})",
                         forLoop.type,
                         tiledVar,
                         forLoop.range.start,
                         tiledVar,
                         forLoop.range.end,
                         tiledVar,
                         params->tileSize)
                   .value();  // shouldn't fail

    ++openedScopeCounter;
    return res + " {";  // Open new scope (Note: after line unlike @outer and @inner)
}

// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); ++i)
std::string regularLoopIdxLineSecond(const LoopMetadata& forLoop,
                                     const AttributedLoop& regularLoop,
                                     const TileParams* params,
                                     int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto blockSize = std::to_string(params->tileSize);

    std::string res;
    if (forLoop.isUnary()) {
        res = util::fmt("for({} {} = {}; {} < ({} + {}); ++{})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        tiledVar,
                        blockSize,
                        forLoop.name)
                  .value();
    } else {
        res = util::fmt("for({} {} = {}; {} < ({} + {}); {} += {})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        tiledVar,
                        blockSize,
                        forLoop.name,
                        forLoop.inc.val)
                  .value();
    }
    return res;
}

std::string getLoopIdxLine(const LoopMetadata& forLoop,
                           const TileParams* params,
                           const LoopOrder& ord,
                           int& openedScopeCounter) {
    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<std::tuple<LoopType, LoopOrder>,
                    std::function<std::string(
                        const LoopMetadata&, const AttributedLoop&, const TileParams*, int&)>>
        mapping{
            {{LoopType::Inner, LoopOrder::First}, innerOuterLoopIdxLineFirst},
            {{LoopType::Outer, LoopOrder::First}, innerOuterLoopIdxLineFirst},
            {{LoopType::Regular, LoopOrder::First}, regularLoopIdxLineFirst},
            {{LoopType::Inner, LoopOrder::Second}, innerOuterLoopIdxLineSecond},
            {{LoopType::Outer, LoopOrder::Second}, innerOuterLoopIdxLineSecond},
            {{LoopType::Regular, LoopOrder::Second}, regularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params, openedScopeCounter);
}

std::string getCheckLine(const LoopMetadata& forLoopMetaData,
                         const TileParams* tileParams,
                         int& openedScopeCounter) {
    if (!tileParams->check) {
        return "";
    }

    // TODO: parse cmp operator
    auto res = util::fmt("if ({} < {})", forLoopMetaData.name, forLoopMetaData.range.end).value();
    return res;
}

// TODO: add check handling
std::string buildPreffixTiledCode(const LoopMetadata& forLoopMetaData,
                                  const TileParams* tileParams,
                                  int& openedScopeCounter) {
    std::string res;
    res += getLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::First, openedScopeCounter);
    res += getLoopIdxLine(forLoopMetaData, tileParams, LoopOrder::Second, openedScopeCounter);
    res += getCheckLine(forLoopMetaData, tileParams, openedScopeCounter);
    return res;
}

std::string buildSuffixTiledCode(int& openedScopeCounter) {
    std::string res;
    // Close all opened scopes
    while (openedScopeCounter--) {
        res += "}";
    }
    return res;
}

}  // namespace

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
    auto usrCtxKey = util::pointerToStr(static_cast<const void*>(a));
    auto tileParams = std::any_cast<TileParams>(s.getUserCtx(usrCtxKey));
    if (tileParams == nullptr) {
        s.pushError(std::error_code(), "No tile params in user context");
        return false;
    }

    auto& astCtx = s.getCompiler().getASTContext();

    if (!isa<ForStmt>(d)) {
        s.pushError(std::error_code(), "Tile can be applied to only for loop");
        return false;
    }
    const auto* forStmt = dyn_cast<ForStmt>(d);
    auto forLoopMetaData = ParseForStmt(const_cast<ForStmt*>(forStmt), astCtx);

    int openedScopeCounter = 0;
    auto prefixCode = buildPreffixTiledCode(forLoopMetaData, tileParams, openedScopeCounter);
    auto suffixCode = buildSuffixTiledCode(openedScopeCounter);

    auto& rewriter = s.getRewriter();

    // Remove attribute + for loop
    SourceRange range;
    range.setBegin(a->getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    range.setEnd(forStmt->getRParenLoc());
    rewriter.RemoveText(range);

    // Insert preffix
    rewriter.InsertText(forStmt->getRParenLoc(), prefixCode);

    // Insert suffix
    rewriter.InsertText(forStmt->getEndLoc(),
                        suffixCode);  // TODO: seems to not work correclty for for loop without {}

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle Tile. Parsed for loop: Init("
                 << "type: " << forLoopMetaData.type << ", name: " << forLoopMetaData.name
                 << ", initValue: " << forLoopMetaData.range.start
                 << "), Cond(rhsExpr: " << forLoopMetaData.range.end
                 << "), Inc(rhsInc: " << forLoopMetaData.inc.val
                 << ", isUnary: " << forLoopMetaData.isUnary() << ")\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
