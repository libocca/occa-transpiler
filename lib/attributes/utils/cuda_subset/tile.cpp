// #include <clang/AST/Decl.h>
// #include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>
// #include <oklt/attributes/frontend/parsers/tile.h>
// #include <oklt/core/attribute_manager/attribute_manager.h>
// #include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/util/string_utils.h>
#include <functional>
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"
#include "handle.h"
#include "attributes/frontend/params/tile.h"

#include <clang/AST/Decl.h>
namespace oklt::cuda_subset {
using namespace clang;
namespace {
const std::string SPACE = " ";
struct ForLoop {
    struct Init {
        const VarDecl* initDecl;
        std::string type;
        std::string name;
        std::string initialValue;
    } init;
    // TODO: add comp operator
    struct Cond {
        const BinaryOperator* condOp;
        std::string rhsExpr;
    } cond;
    struct Inc {
        const Expr* incExpr;
        std::string rhsInc;
        bool isUnary;
    } inc;
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

tl::expected<ForLoop, Error> parseForLoopInit(ForLoop& forLoop,
                                              const clang::Stmt* init,
                                              ASTContext& astCtx) {
    Error err{std::error_code(), "Init part of for statement must be variable declaration"};
    const DeclStmt* initDecl;
    if (!(initDecl = dyn_cast_or_null<DeclStmt>(init)) || !(initDecl->isSingleDecl())) {
        return tl::make_unexpected(err);
    }

    const VarDecl* varDecl;
    if (!(varDecl = dyn_cast<VarDecl>(initDecl->getSingleDecl()))) {
        return tl::make_unexpected(err);
    }

    const Expr* start = varDecl->getInit();
    while (auto rhs = dyn_cast_or_null<CastExpr>(start)) {
        start = rhs->getSubExpr();
    }

    forLoop.init = ForLoop::Init{
        .initDecl = varDecl,
        .type = varDecl->getType().getAsString(),
        .name = varDecl->getDeclName().getAsString(),
        .initialValue = prettyPrint(start, astCtx.getPrintingPolicy()),
    };
    return forLoop;
}

tl::expected<ForLoop, Error> parseForLoopCond(ForLoop& forLoop,
                                              const clang::Expr* cond,
                                              ASTContext& astCtx) {
    Error err{std::error_code(), "Condition must be a comparisson"};

    const BinaryOperator* binOp;
    if (!(binOp = dyn_cast_or_null<BinaryOperator>(cond)) || !binOp->isComparisonOp()) {
        return tl::make_unexpected(err);
    }

    auto rhs = dyn_cast_or_null<Expr>(binOp->getRHS());
    // while (rhs & rhs->getSubExpr() && isa<CastExpr>(rhs->getSubExpr())) {
    //     rhs = dyn_cast_or_null<CastExpr>(rhs->getSubExpr());
    // };

    if (!rhs) {
        llvm::outs() << "hello\n";
        return tl::make_unexpected(err);
    }

    forLoop.cond = ForLoop::Cond{
        .condOp = binOp,
        .rhsExpr = prettyPrint(rhs, astCtx.getPrintingPolicy()),
    };

    return forLoop;
}

tl::expected<ForLoop, Error> parseForLoopInc(ForLoop& forLoop,
                                             const clang::Expr* inc,
                                             ASTContext& astCtx) {
    Error err{std::error_code(),
              "Increment should be unary or compound, using initialized variable"};
    if (isa<UnaryOperator>(inc)) {
        forLoop.inc = ForLoop::Inc{.incExpr = inc, .rhsInc = "1", .isUnary = true};
    } else if (isa<CompoundAssignOperator>(inc)) {
        auto incCompound = dyn_cast<CompoundAssignOperator>(inc);
        auto lhs = dyn_cast_or_null<DeclRefExpr>(incCompound->getLHS());
        if (lhs && lhs->getNameInfo().getAsString() != forLoop.init.name) {
            return tl::make_unexpected(err);
        }
        forLoop.inc =
            ForLoop::Inc{.incExpr = inc,
                         .rhsInc = prettyPrint(incCompound->getRHS(), astCtx.getPrintingPolicy()),
                         .isUnary = false};

    } else {
        return tl::make_unexpected(err);
    }

    return forLoop;
}

// TODO: Use same parsing with other parts of code in the future
tl::expected<ForLoop, Error> parseForLoop(const clang::ForStmt* forStmt, ASTContext& astCtx) {
    ForLoop ret;
    return parseForLoopInit(ret, forStmt->getInit(), astCtx)
        .and_then([&](const auto& _) { return parseForLoopCond(ret, forStmt->getCond(), astCtx); })
        .and_then([&](const auto& _) { return parseForLoopInc(ret, forStmt->getInc(), astCtx); });
}

std::string dimToStr(const Dim& dim) {
    static std::map<Dim, std::string> mapping{{Dim::X, "x"}, {Dim::Y, "y"}, {Dim::Z, "z"}};
    return mapping[dim];
}

std::string getTiledVariableName(const ForLoop& forLoop) {
    return "_occa_tiled_" + forLoop.init.name;
}

std::string scopeWrap(const std::string& content) {
    return "{\n" + content + "\n}";
}

// TODO: fix code dublication
// Produces something like: int i = _occa_tiled_i + (inc * threadIdx.x);
std::string innerLoopIdxLineSecond(const ForLoop& forLoop,
                                   const Loop& innerLoop,
                                   const TileParams* params) {
    static_cast<void>(params);
    std::vector<std::string> parts{
        forLoop.init.type,
        forLoop.init.name,
        "=",
        getTiledVariableName(forLoop),
        "+",
    };

    std::string threadIdx = "threadIdx." + dimToStr(innerLoop.dim);
    if (forLoop.inc.isUnary) {
        parts.push_back(threadIdx);
    } else {
        parts.push_back("((" + forLoop.inc.rhsInc + ") * " + threadIdx + ");");
    }
    auto res = util::join(parts.begin(), parts.end(), SPACE);
    return "{\n" + res + "\n";  // Open new scope
}

// Produces something like: int _occa_tiled_i = init + ((tileSize * inc) * threadIdx.x);
std::string innerLoopIdxLineFirst(const ForLoop& forLoop,
                                  const Loop& outerLoop,
                                  const TileParams* params) {
    auto blockSize = std::to_string(params->tileSize);
    auto innerLoopSize = forLoop.inc.rhsInc;
    auto blockIdx = "threadIdx." + dimToStr(outerLoop.dim);

    std::vector<std::string> parts{
        forLoop.init.type,
        getTiledVariableName(forLoop),
        "=",
        forLoop.init.initialValue,
        "+",
        "((" + blockSize + " * " + innerLoopSize + ") * " + blockIdx + ");",
    };
    auto res = util::join(parts.begin(), parts.end(), SPACE);
    return "{\n" + res + "\n";  // Open new scope
}

// Produces something like: int _occa_tiled_i = init + ((tileSize * inc) * blockIdx.x);
std::string outerLoopIdxLineFirst(const ForLoop& forLoop,
                                  const Loop& outerLoop,
                                  const TileParams* params) {
    auto blockSize = std::to_string(params->tileSize);
    auto innerLoopSize = forLoop.inc.rhsInc;
    auto blockIdx = "blockIdx." + dimToStr(outerLoop.dim);

    std::vector<std::string> parts{
        forLoop.init.type,
        getTiledVariableName(forLoop),
        "=",
        forLoop.init.initialValue,
        "+",
        "((" + blockSize + " * " + innerLoopSize + ") * " + blockIdx + ");",
    };
    auto res = util::join(parts.begin(), parts.end(), SPACE);
    return "{\n" + res + "\n";  // Open new scope
}

// Produces something like: int i = _occa_tiled_i + (inc * blockIdx.x);
std::string outerLoopIdxLineSecond(const ForLoop& forLoop,
                                   const Loop& innerLoop,
                                   const TileParams* params) {
    static_cast<void>(params);
    std::vector<std::string> parts{
        forLoop.init.type,
        forLoop.init.name,
        "=",
        getTiledVariableName(forLoop),
        "+",
    };

    std::string threadIdx = "blockIdx." + dimToStr(innerLoop.dim);
    if (forLoop.inc.isUnary) {
        parts.push_back(threadIdx);
    } else {
        parts.push_back("((" + forLoop.inc.rhsInc + ") * " + threadIdx + ");");
    }
    auto res = util::join(parts.begin(), parts.end(), SPACE);
    return "{\n" + res + "\n";  // Open new scope
}

// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); ++i)
std::string regularLoopIdxLineFirst(const ForLoop& forLoop,
                                    const Loop& regularLoop,
                                    const TileParams* params) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto blockSize = std::to_string(params->tileSize);

    std::vector<std::string> parts{
        "for",
        "(" + tiledVar,
        forLoop.init.name,
        "=",
        forLoop.init.initialValue + ";",
        tiledVar,
        "<",  // TODO: parse cmp operator
        forLoop.cond.rhsExpr + ";",
        tiledVar,
        "+=",  // TODO: Are other operators possible?
        std::to_string(params->tileSize),
        ")",
    };

    auto res = util::join(parts.begin(), parts.end(), SPACE);
    return res + " {\n";  // Open new scope (Note: after line unlike @outer and @inner)
}

// Produces something like: for (int i = _occa_tiled_i; i < (_occa_tiled_i + tileSize); ++i)
std::string regularLoopIdxLineSecond(const ForLoop& forLoop,
                                     const Loop& regularLoop,
                                     const TileParams* params) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto blockSize = std::to_string(params->tileSize);

    std::vector<std::string> parts{
        "for",
        "(" + forLoop.init.type,
        forLoop.init.name,
        "=",
        tiledVar + ";",
        forLoop.init.name,
        "<",  // TODO: parse cmp operator
        "(" + tiledVar,
        "+",
        blockSize + ");",
        "++" + forLoop.init.name + ")",
    };

    auto res = util::join(parts.begin(), parts.end(), SPACE);
    return res + " {\n";  // Open new scope (Note: after line unlike @outer and @inner)
}

std::string getLoopIdxLine(const ForLoop& forLoop, const TileParams* params, const LoopOrder& ord) {
    // TODO: this logic should be based on first or second loop, not inner/outer/regular
    static std::map<std::tuple<LoopType, LoopOrder>, std::function<std::string(const ForLoop&, const Loop&, const TileParams*)>>
        mapping{
            {{LoopType::Inner, LoopOrder::First}, innerLoopIdxLineFirst},
            {{LoopType::Outer, LoopOrder::First}, outerLoopIdxLineFirst},
            {{LoopType::Regular, LoopOrder::First}, regularLoopIdxLineFirst},
            {{LoopType::Inner, LoopOrder::Second}, innerLoopIdxLineSecond},
            {{LoopType::Outer, LoopOrder::Second}, outerLoopIdxLineSecond},
            {{LoopType::Regular, LoopOrder::Second}, regularLoopIdxLineSecond},
        };
    auto& loop = ord == LoopOrder::First ? params->firstLoop : params->secondLoop;
    return mapping[{loop.type, ord}](forLoop, loop, params);
}

std::string getCheckLine(const ForLoop& forLoop, const TileParams* tileParams) {
    if (!tileParams->check) {
        return "";
    }

    // TODO: parse cmp operator
    auto res = "if (" + forLoop.init.name + " < " + forLoop.cond.rhsExpr + ")";
    return res + " {\n";  // Open new scope
}

// TODO: add check handling
std::string buildPreffixTiledCode(const ForLoop& forLoop, const TileParams* tileParams) {
    std::string res;
    res += getLoopIdxLine(forLoop, tileParams, LoopOrder::First);
    res += getLoopIdxLine(forLoop, tileParams, LoopOrder::Second);
    res += getCheckLine(forLoop, tileParams);
    return res;
}

std::string buildSuffixTiledCode(const ForLoop& forLoop, const TileParams* tileParams) {
    std::string res;
    res += "}\n";  // Close scope created by inner loop
    if (tileParams->check) {
        res += "}\n";  // Close scope created by check
    }
    return res;
}

}  // namespace

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
    auto usrCtxKey = util::pointerToStr(static_cast<const void*>(a));
    auto tileParams = std::any_cast<TileParams>(s.getUserCtx(usrCtxKey));
    if (tileParams == nullptr) {
        s.pushError(std::error_code(), "no tile params in user context");
        return false;
    }

    auto& astCtx = s.getCompiler().getASTContext();

    if (!isa<ForStmt>(d)) {
        s.pushError(std::error_code(), "Tile can be applied to only for loop");
        return false;
    }
    const auto* forStmt = dyn_cast<ForStmt>(d);
    auto forLoop = parseForLoop(forStmt, astCtx);
    if (!forLoop) {
        s.pushError(forLoop.error());
        return false;
    }

    auto prefixCode = buildPreffixTiledCode(forLoop.value(), tileParams);
    auto suffixCode = buildSuffixTiledCode(forLoop.value(), tileParams);

    auto& rewriter = s.getRewriter();

    // Remove attribute + for loop
    SourceRange range;
    range.setBegin(a->getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    range.setEnd(forStmt->getRParenLoc());
    rewriter.RemoveText(range);

    // Insert preffix
    rewriter.InsertText(forStmt->getRParenLoc(), prefixCode);

    // Insert suffix
    rewriter.InsertText(forStmt->getEndLoc(), suffixCode);

#ifdef TRANSPILER_DEBUG_LOG
    llvm::outs() << "[DEBUG] Handle Tile. Parsed for loop: Init("
                 << "type: " << forLoop->init.type << ", name: " << forLoop->init.name
                 << ", initValue: " << forLoop->init.initialValue
                 << "), Cond(rhsExpr: " << forLoop->cond.rhsExpr
                 << "), Inc(rhsInc: " << forLoop->inc.rhsInc
                 << ", isUnary: " << forLoop->inc.isUnary << ")\n";
#endif
    return true;
}
}  // namespace oklt::cuda_subset
