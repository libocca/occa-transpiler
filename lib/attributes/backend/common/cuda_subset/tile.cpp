#include <clang/AST/Decl.h>
#include <oklt/attributes/backend/common/cuda_subset/cuda_subset.h>
#include <oklt/core/attribute_manager/attribute_manager.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/util/string_utils.h>
#include <oklt/attributes/frontend/parsers/tile.hpp>

namespace oklt::cuda_subset {
using namespace clang;

struct ForLoop {
    struct Init {
        const VarDecl* initDecl;
        std::string type;
        std::string name;
        std::string initialValue;
    } init;
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

    auto rhs = dyn_cast_or_null<CastExpr>(binOp->getRHS());
    while (rhs && rhs->getSubExpr() && isa<CastExpr>(rhs->getSubExpr())) {
        rhs = dyn_cast_or_null<CastExpr>(rhs->getSubExpr());
    };

    if (!rhs) {
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

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s) {
    auto usrCtxKey = util::pointerToStr(static_cast<const void*>(a));
    auto tileParams = std::any_cast<TileParams>(s.getUserCtx(usrCtxKey));

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
