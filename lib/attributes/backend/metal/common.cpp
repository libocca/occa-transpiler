#include "attributes/backend/metal/common.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/utils/range_to_string.h"
#include "util/string_utils.hpp"

#include <clang/Rewrite/Core/Rewriter.h>

namespace oklt::metal {
using namespace clang;

std::string axisToStr(const Axis& axis) {
    static std::map<Axis, std::string> mapping{{Axis::X, "x"}, {Axis::Y, "y"}, {Axis::Z, "z"}};
    return mapping[axis];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strAxis = axisToStr(loop.axis);
    switch (loop.type) {
        case (LoopType::Inner):
            return util::fmt("_occa_thread_position.{}", strAxis).value();
        case (LoopType::Outer):
            return util::fmt(" _occa_group_position.{}", strAxis).value();
        default:  // Incorrect case
            return "";
    }
}

std::string getTiledVariableName(const OklLoopInfo& forLoop) {
    return "_occa_tiled_" + forLoop.var.name;
}

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

}  // namespace oklt::metal
