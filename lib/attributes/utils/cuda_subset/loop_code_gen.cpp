#include "loop_code_gen.h"
#include <oklt/util/string_utils.h>

namespace oklt::cuda_subset {
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

std::string getCondCompStr(const LoopMetadata& forLoop) {
    switch (forLoop.condition.op) {
        case clang::BO_LE:
            return "<=";
        case clang::BO_LT:
            return "<";
        case clang::BO_GE:
            return ">=";
        case clang::BO_GT:
            return ">";
        default:  // Shouldn't happen, since for loop parse validates operator
            return "<error>";
    }
}

std::string getUnaryStr(const LoopMetadata& forLoop, const std::string& var) {
    switch (forLoop.inc.op.uo) {
        case clang::UO_PreInc:
            return "++" + var;
        case clang::UO_PostInc:
            return var + "++";
        case clang::UO_PreDec:
            return "--" + var;
        case clang::UO_PostDec:
            return var + "--";

        default:  // Shouldn't happen, since for loop parse validates operator
            return "<error>";
    }
}

std::string buildCloseScopes(int& openedScopeCounter) {
    std::string res;
    // Close all opened scopes
    while (openedScopeCounter--) {
        res += "}";
    }
    return res;
}

void replaceAttributedLoop(const clang::Attr* a,
                           const clang::ForStmt* f,
                           const std::string& prefixCode,
                           const std::string& suffixCode,
                           SessionStage& s) {
    auto& rewriter = s.getRewriter();

    // Remove attribute + for loop:
    //      @attribute(...) for (int i = start; i < end; i += inc)
    //  or: for (int i = start; i < end; i += inc; @attribute(...))
    clang::SourceRange range;
    range.setBegin(a->getRange().getBegin().getLocWithOffset(-2));  // TODO: remove magic number
    range.setEnd(f->getRParenLoc());
    rewriter.RemoveText(range);

    // Insert preffix
    rewriter.InsertText(f->getRParenLoc(), prefixCode);

    // Insert suffix
    rewriter.InsertText(f->getEndLoc(),
                        suffixCode);  // TODO: seems to not work correclty for for loop without {}
}

namespace tile {
std::string getTiledVariableName(const LoopMetadata& forLoop) {
    return "_occa_tiled_" + forLoop.name;
}

std::string buildIinnerOuterLoopIdxLineFirst(const LoopMetadata& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} ({} * {});",
                                  forLoop.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} (({} * {}) * {});",
                                  forLoop.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  op,
                                  params->tileSize,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    return res;
}

std::string buildInnerOuterLoopIdxLineSecond(const LoopMetadata& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    static_cast<void>(params);
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(
            util::fmt("{} {} = {} {} {};", forLoop.type, forLoop.name, tiledVar, op, idx).value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  forLoop.type,
                                  forLoop.name,
                                  tiledVar,
                                  op,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return "{" + res;  // Open new scope
}

std::string buildRegularLoopIdxLineFirst(const LoopMetadata& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto blockSize = std::to_string(params->tileSize);
    auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(forLoop);

    auto res = util::fmt("for({} {} = {}; {} {} {}; {} {} {})",
                         forLoop.type,
                         tiledVar,
                         forLoop.range.start,
                         tiledVar,
                         cmpOpStr,
                         forLoop.range.end,
                         tiledVar,
                         assignUpdate,
                         params->tileSize)
                   .value();  // shouldn't fail

    ++openedScopeCounter;
    return res + " {";  // Open new scope (Note: after line unlike @outer and @inner)
}

std::string buildRegularLoopIdxLineSecond(const LoopMetadata& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto blockSize = std::to_string(params->tileSize);
    auto op = forLoop.IsInc() ? "+" : "-";
    auto cmp = forLoop.IsInc() ? "<" : ">";

    std::string res;
    if (forLoop.isUnary()) {
        auto unaryStr = getUnaryStr(forLoop, forLoop.name);  // ++i/i++/--i/i--
        res = util::fmt("for({} {} = {}; {} {} ({} {} {}); {})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        cmp,
                        tiledVar,
                        op,
                        blockSize,
                        unaryStr)
                  .value();
    } else {
        auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
        res = util::fmt("for({} {} = {}; {} {} ({} {} {}); {} {} {})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        cmp,
                        tiledVar,
                        op,
                        blockSize,
                        forLoop.name,
                        assignUpdate,
                        forLoop.inc.val)
                  .value();
    }
    return res;
}
}  // namespace tile

namespace inner_outer {
std::string buildInnerOuterLoopIdxLine(const LoopMetadata& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter) {
    static_cast<void>(openedScopeCounter);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(
            util::fmt("{} {} = {} {} {};", forLoop.type, forLoop.name, forLoop.range.start, op, idx)
                .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  forLoop.type,
                                  forLoop.name,
                                  forLoop.range.start,
                                  op,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    return res;  // Open new scope
}

}  // namespace inner_outer

}  // namespace oklt::cuda_subset