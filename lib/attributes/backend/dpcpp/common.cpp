#include "attributes/backend/dpcpp/common.h"
#include <oklt/util/string_utils.h>
#include "core/sema/okl_sema_ctx.h"

namespace oklt::dpcpp {

std::string axisToStr(const Axis& dim) {
    // TODO: Verify that this is a correct mapping from original OKL transpiler developera
    //      (intuitively should be x->0, y->1, z->2)
    static std::map<Axis, std::string> mapping{
        {Axis::X, "2"}, {Axis::Y, "1"}, {Axis::Z, "0"}};
    return mapping[dim];
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
}  // namespace oklt::dpcpp
