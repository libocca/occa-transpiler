#include <oklt/util/string_utils.h>

#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"

namespace oklt::cuda_subset {
std::string axisToStr(const Axis& dim) {
    static std::map<Axis, std::string> mapping{
        {Axis::X, "x"}, {Axis::Y, "y"}, {Axis::Z, "z"}};
    return mapping[dim];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strDim = axisToStr(loop.axis);
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
    auto& meta = forLoop.metadata;
    return "_occa_tiled_" + meta.var.name;
}

std::string buildIinnerOuterLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);
    auto& meta = forLoop.metadata;
    auto op = meta.IsInc() ? "+" : "-";

    std::string res;
    if (meta.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});",
                                  meta.var.type,
                                  tiledVar,
                                  meta.range.start,
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} ((({}) * {}) * {});",
                                  meta.var.type,
                                  tiledVar,
                                  meta.range.start,
                                  op,
                                  params->tileSize,
                                  meta.inc.val,
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
    auto& meta = forLoop.metadata;
    auto op = meta.IsInc() ? "+" : "-";

    std::string res;
    if (meta.isUnary()) {
        res = std::move(
            util::fmt("{} {} = {} {} {};", meta.var.type, meta.var.name, tiledVar, op, idx)
                .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  meta.var.type,
                                  meta.var.name,
                                  tiledVar,
                                  op,
                                  meta.inc.val,
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
    auto& meta = forLoop.metadata;
    auto assignUpdate = meta.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(meta.condition.op);

    auto res = util::fmt("for({} {} = {}; {} {} {}; {} {} ({}))",
                         meta.var.type,
                         tiledVar,
                         meta.range.start,
                         tiledVar,
                         cmpOpStr,
                         meta.range.end,
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
    auto& meta = forLoop.metadata;
    auto op = meta.IsInc() ? "+" : "-";
    auto cmp = meta.IsInc() ? "<" : ">";

    std::string res;
    if (meta.isUnary()) {
        auto unaryStr = getUnaryStr(meta.inc.op.uo, meta.var.name);  // ++i/i++/--i/i--
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {})",
                        meta.var.type,
                        meta.var.name,
                        tiledVar,
                        meta.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        unaryStr)
                  .value();
    } else {
        auto assignUpdate = meta.IsInc() ? "+=" : "-=";
        res = util::fmt("for({} {} = {}; {} {} ({} {} ({})); {} {} {})",
                        meta.var.type,
                        meta.var.name,
                        tiledVar,
                        meta.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        meta.var.name,
                        assignUpdate,
                        meta.inc.val)
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
    auto& meta = forLoop.metadata;
    auto op = meta.IsInc() ? "+" : "-";

    std::string res;
    if (meta.isUnary()) {
        res = std::move(
            util::fmt("{} {} = {} {} {};", meta.var.type, meta.var.name, meta.range.start, op, idx)
                .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  meta.var.type,
                                  meta.var.name,
                                  meta.range.start,
                                  op,
                                  meta.inc.val,
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
