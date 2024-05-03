#include "util/string_utils.hpp"

#include "attributes/utils/code_gen.h"
#include "attributes/utils/cuda_subset/loop_code_gen.h"
#include "core/sema/okl_sema_info.h"
#include "core/utils/range_to_string.h"

#include <clang/Rewrite/Core/Rewriter.h>

namespace oklt::cuda_subset {
std::string axisToStr(const Axis& axis) {
    static std::map<Axis, std::string> mapping{{Axis::X, "x"}, {Axis::Y, "y"}, {Axis::Z, "z"}};
    return mapping[axis];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strAxis = axisToStr(loop.axis);
    switch (loop.type) {
        case (LoopType::Inner):
            return util::fmt("threadIdx.{}", strAxis).value();
        case (LoopType::Outer):
            return util::fmt("blockIdx.{}", strAxis).value();
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
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});\n",
                                  forLoop.var.typeName,
                                  tiledVar,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  params->tileSize,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} ((({}) * {}) * {});\n",
                                  forLoop.var.typeName,
                                  tiledVar,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  params->tileSize,
                                  getLatestSourceText(forLoop.inc.val, rewriter),
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return " {\n" + res;
}

std::string buildInnerOuterLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                             const AttributedLoop& loop,
                                             const TileParams* params,
                                             int& openedScopeCounter,
                                             oklt::Rewriter& rewriter) {
    static_cast<void>(params);
    auto tiledVar = getTiledVariableName(forLoop);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(
            util::fmt(
                "{} {} = {} {} {};\n", forLoop.var.typeName, forLoop.var.name, tiledVar, op, idx)
                .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});\n",
                                  forLoop.var.typeName,
                                  forLoop.var.name,
                                  tiledVar,
                                  op,
                                  getLatestSourceText(forLoop.inc.val, rewriter),
                                  idx)
                            .value());
    }
    ++openedScopeCounter;
    return " {\n" + res;  // Open new scope
}

std::string buildRegularLoopIdxLineFirst(const OklLoopInfo& forLoop,
                                         const AttributedLoop& regularLoop,
                                         const TileParams* params,
                                         int& openedScopeCounter,
                                         oklt::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto assignUpdate = forLoop.IsInc() ? "+=" : "-=";
    auto cmpOpStr = getCondCompStr(forLoop.condition.op);

    auto res = util::fmt("for ({} {} = {}; {} {} {}; {} {} ({}))",
                         forLoop.var.typeName,
                         tiledVar,
                         getLatestSourceText(forLoop.range.start, rewriter),
                         tiledVar,
                         cmpOpStr,
                         getLatestSourceText(forLoop.range.end, rewriter),
                         tiledVar,
                         assignUpdate,
                         params->tileSize)
                   .value();  // shouldn't fail

    ++openedScopeCounter;
    return res + " {\n";  // Open new scope (Note: after line unlike @outer and @inner)
}

std::string buildRegularLoopIdxLineSecond(const OklLoopInfo& forLoop,
                                          const AttributedLoop& regularLoop,
                                          const TileParams* params,
                                          int& openedScopeCounter,
                                          oklt::Rewriter& rewriter) {
    auto tiledVar = getTiledVariableName(forLoop);
    auto& stmt = forLoop.stmt;
    auto op = forLoop.IsInc() ? "+" : "-";
    auto cmp = forLoop.IsInc() ? "<" : ">";

    std::string res;
    if (forLoop.isUnary()) {
        auto unaryStr = getUnaryStr(forLoop.inc.op.uo, forLoop.var.name);  // ++i/i++/--i/i--
        res = util::fmt("for ({} {} = {}; {} {} ({} {} ({})); {})",
                        forLoop.var.typeName,
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
        res = util::fmt("for ({} {} = {}; {} {} ({} {} ({})); {} {} {})",
                        forLoop.var.typeName,
                        forLoop.var.name,
                        tiledVar,
                        forLoop.var.name,
                        cmp,
                        tiledVar,
                        op,
                        params->tileSize,
                        forLoop.var.name,
                        assignUpdate,
                        getLatestSourceText(forLoop.inc.val, rewriter))
                  .value();
    }

    if (params->check || !llvm::isa<clang::CompoundStmt>(stmt.getBody())) {
        ++openedScopeCounter;
        res += " {\n";
    }

    return res;
}
}  // namespace tile

namespace inner_outer {
std::string buildInnerOuterLoopIdxLine(const OklLoopInfo& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter,
                                       oklt::Rewriter& rewriter) {
    static_cast<void>(openedScopeCounter);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = ({}) {} {};\n",
                                  forLoop.var.typeName,
                                  forLoop.var.name,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = ({}) {} (({}) * {});\n",
                                  forLoop.var.typeName,
                                  forLoop.var.name,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  getLatestSourceText(forLoop.inc.val, rewriter),
                                  idx)
                            .value());
    }

    return res;
}

}  // namespace inner_outer

}  // namespace oklt::cuda_subset
