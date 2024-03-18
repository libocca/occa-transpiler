#include <oklt/util/string_utils.h>
#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"
#include "tl/expected.hpp"

#include "attributes/utils/tile_utils.h"

namespace oklt {
namespace {
tl::expected<TileParams, Error> updateParamsAxis(TileParams& params,
                                                 OklLoopInfo& loopInfo,
                                                 bool isFirst,
                                                 size_t heightLimit) {
    AttributedLoop& attrLoop = isFirst ? params.firstLoop : params.secondLoop;
    if (attrLoop.axis != Axis::Auto || attrLoop.type == LoopType::Regular) {
        return params;
    }
    auto [metaLoopType, loopName] = [&]() -> std::pair<LoopType, std::string> {
        if (attrLoop.type == LoopType::Inner) {
            return {LoopType::Inner, "@inner"};
        }
        return {LoopType::Outer, "@outer"};
    }();
    auto height = loopInfo.getHeightSameType(metaLoopType);
    // Case of @outer @outer or @inner @inner
    if (isFirst && loopInfo.type[1] == metaLoopType) {
        ++height;
    }
    if (height > heightLimit) {
        return tl::make_unexpected(Error{
            {}, util::fmt("More than {} nested [{}] loops", heightLimit + 1, loopName).value()});
    }
    attrLoop.axis = static_cast<Axis>(height);
    return params;
}
}  // namespace

tl::expected<TileParams, Error> tileParamsHandleAutoAxes(const TileParams& params,
                                                         OklLoopInfo& loopInfo,
                                                         size_t heightLimit) {
    TileParams res = params;
    return updateParamsAxis(res, loopInfo, true, heightLimit).and_then([&](auto params) {
        return updateParamsAxis(res, loopInfo, false, heightLimit);
    });
}
}  // namespace oklt
