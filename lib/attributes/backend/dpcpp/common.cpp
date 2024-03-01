#include "attributes/backend/dpcpp/common.h"
#include <oklt/util/string_utils.h>

namespace oklt::dpcpp {

std::string dimToStr(const Dim& dim) {
    // TODO: Verify that this is a correct mapping from original OKL transpiler developera
    //      (intuitively should be x->0, y->1, z->2)
    static std::map<Dim, std::string> mapping{{Dim::X, "2"}, {Dim::Y, "1"}, {Dim::Z, "0"}};
    return mapping[dim];
}

std::string getIdxVariable(const AttributedLoop& loop) {
    auto strDim = dimToStr(loop.dim);
    switch (loop.type) {
        case (LoopType::Inner):
            return util::fmt("item.get_local_id({})", strDim).value();
        case (LoopType::Outer):
            return util::fmt("item_.get_group({})", strDim).value();
        default:  // Incorrect case
            return "";
    }
}
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
    return res;
}
}  // namespace oklt::dpcpp
