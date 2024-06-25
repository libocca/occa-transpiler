#include "attributes/backend/dpcpp/common.h"
#include "util/string_utils.hpp"
#include "core/sema/okl_sema_ctx.h"
#include "core/utils/range_to_string.h"

#include <clang/Rewrite/Core/Rewriter.h>

#include <spdlog/spdlog.h>

namespace oklt::opencl {
using namespace clang;

std::string axisToStr(const Axis& axis) {
    static std::map<Axis, std::string> mapping{{Axis::X, "0"}, {Axis::Y, "1"}, {Axis::Z, "2"}};
    return mapping[axis];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strAxis = axisToStr(loop.axis);
    switch (loop.type) {
        case (LoopType::Inner):
            return util::fmt("get_local_id({})", strAxis).value();
        case (LoopType::Outer):
            return util::fmt("get_group_id({})", strAxis).value();
        default:  // Incorrect case
            return "";
    }
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

}
