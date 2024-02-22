#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include <oklt/util/string_utils.h>
#include "attributes/utils/code_gen.h"

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
std::string getTiledVariableName(const LoopMetaData& forLoop) {
    return "_occa_tiled_" + forLoop.name;
}

std::string buildIinnerOuterLoopIdxLineFirst(const LoopMetaData& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});",
                                  forLoop.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} ((({}) * {}) * {});",
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

std::string buildInnerOuterLoopIdxLineSecond(const LoopMetaData& forLoop,
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

std::string buildRegularLoopIdxLineFirst(const LoopMetaData& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(forLoop.condition.op);

    auto res = util::fmt("for({} {} = {}; {} {} {}; {} {} ({}))",
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

std::string buildRegularLoopIdxLineSecond(const LoopMetaData& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto op = forLoop.IsInc() ? "+" : "-";
    auto cmp = forLoop.IsInc() ? "<" : ">";

    std::string res;
    if (forLoop.isUnary()) {
        auto unaryStr = getUnaryStr(forLoop.inc.op.uo, forLoop.name);  // ++i/i++/--i/i--
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        unaryStr)
                  .value();
    } else {
        auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {} {} {})",
                        forLoop.type,
                        forLoop.name,
                        tiledVar,
                        forLoop.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        forLoop.name,
                        assignUpdate,
                        forLoop.inc.val)
                  .value();
    }
    return res;
}
}  // namespace tile

namespace inner_outer {
std::string buildInnerOuterLoopIdxLine(const LoopMetaData& forLoop,
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
