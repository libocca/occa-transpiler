#include <oklt/util/string_utils.h>

#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "core/sema/okl_sema_info.h"

namespace oklt::cuda_subset {
std::string dimToStr(const Axis& dim) {
    static std::map<Axis, std::string> mapping{{Axis::X, "x"}, {Axis::Y, "y"}, {Axis::Z, "z"}};
    return mapping[dim];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strDim = dimToStr(loop.axis);
    switch (loop.type) {
        case (LoopType::Inner):
            return util::fmt("threadIdx.{}", strDim).value();
        case (LoopType::Outer):
            return util::fmt("blockIdx.{}", strDim).value();
        default:  // Incorrect case
            return "";
    }
}

namespace tile {
std::string getTiledVariableName(const OklLoopInfo& forLoop) {
    return "_occa_tiled_" + forLoop.var.name;
}

std::string buildIinnerOuterLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});",
                                  forLoop.var.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} ((({}) * {}) * {});",
                                  forLoop.var.type,
                                  tiledVar,
                                  forLoop.range.start,
                                  op,
                                  params->tileSize,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return "{" + res;
}

std::string buildInnerOuterLoopIdxLineSecond(const OklLoopInfo& forLoop,
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
            util::fmt("{} {} = {} {} {};", forLoop.var.type, forLoop.var.name, tiledVar, op, idx)
                .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  forLoop.var.type,
                                  forLoop.var.name,
                                  tiledVar,
                                  op,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return "{" + res;  // Open new scope
}

std::string buildRegularLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(forLoop.condition.op);

    auto res = util::fmt("for({} {} = {}; {} {} {}; {} {} ({}))",
                         forLoop.var.type,
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

std::string buildRegularLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto op = forLoop.IsInc() ? "+" : "-";
    auto cmp = forLoop.IsInc() ? "<" : ">";

    std::string res;
    if (forLoop.isUnary()) {
        auto unaryStr = getUnaryStr(forLoop.inc.op.uo, forLoop.var.name);  // ++i/i++/--i/i--
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {})",
                        forLoop.var.type,
                        forLoop.var.name,
                        tiledVar,
                        forLoop.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        unaryStr)
                  .value();
    } else {
        auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {} {} {})",
                        forLoop.var.type,
                        forLoop.var.name,
                        tiledVar,
                        forLoop.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        forLoop.var.name,
                        assignUpdate,
                        forLoop.inc.val)
                  .value();
    }
    return res;
}
}  // namespace tile

namespace inner_outer {
std::string buildInnerOuterLoopIdxLine(const OklLoopInfo& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter) {
    static_cast<void>(openedScopeCounter);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = {} {} {};",
                                  forLoop.var.type,
                                  forLoop.var.name,
                                  forLoop.range.start,
                                  op,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  forLoop.var.type,
                                  forLoop.var.name,
                                  forLoop.range.start,
                                  op,
                                  forLoop.inc.val,
                                  idx)
                            .value());
    }
    if (loop.type == LoopType::Outer) {
        return res;
    }
    ++openedScopeCounter;
    return "{" + res;
}

}  // namespace inner_outer

}  // namespace oklt::cuda_subset
