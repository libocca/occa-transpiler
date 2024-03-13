#include "attributes/backend/dpcpp/common.h"
#include <oklt/util/string_utils.h>
#include "clang/Rewrite/Core/Rewriter.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/utils/range_to_string.h"

namespace oklt::dpcpp {

std::string axisToStr(const Axis& axis) {
    // TODO: Verify that this is a correct mapping from original OKL transpiler developera
    //      (intuitively should be x->0, y->1, z->2)
    static std::map<Axis, std::string> mapping{
        {Axis::X, "2"}, {Axis::Y, "1"}, {Axis::Z, "0"}};
    return mapping[axis];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strAxis = axisToStr(loop.axis);
    switch (loop.type) {
        case (LoopType::Inner):
            return util::fmt("item.get_local_id({})", strAxis).value();
        case (LoopType::Outer):
            return util::fmt("item_.get_group({})", strAxis).value();
        default:  // Incorrect case
            return "";
    }
}
std::string buildInnerOuterLoopIdxLine(const OklLoopInfo& forLoop,
                                       const AttributedLoop& loop,
                                       int& openedScopeCounter,
                                       clang::Rewriter& rewriter) {
    static_cast<void>(openedScopeCounter);
    auto idx = getIdxVariable(loop);
    auto op = forLoop.IsInc() ? "+" : "-";

    std::string res;
    if (forLoop.isUnary()) {
        res = std::move(util::fmt("{} {} = {} {} {};",
                                  forLoop.var.typeName,
                                  forLoop.var.name,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  idx)
                            .value());
    } else {
        res = std::move(util::fmt("{} {} = {} {} (({}) * {});",
                                  forLoop.var.typeName,
                                  forLoop.var.name,
                                  getLatestSourceText(forLoop.range.start, rewriter),
                                  op,
                                  getLatestSourceText(forLoop.inc.val, rewriter),
                                  idx)
                            .value());
    }
    if (loop.type == LoopType::Outer) {
        return res;
    }
    ++openedScopeCounter;
    return "{" + res;
}
}  // namespace oklt::dpcpp
