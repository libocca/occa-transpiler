#include <oklt/util/string_utils.h>
#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"
#include "tl/expected.hpp"

#include "attributes/utils/tile_utils.h"

namespace oklt {
namespace {
tl::expected<TileParams, Error> updateParamsDim(TileParams& params,
                                                OklLoopInfo& loopInfo,
                                                bool isFirst,
                                                size_t heightLimit) {
    AttributedLoop& attrLoop = isFirst ? params.firstLoop : params.secondLoop;
    if (attrLoop.dim != DimType::Auto || attrLoop.type == AttributedLoopType::Regular) {
        return params;
    }
    auto [metaLoopType, loopName] = [&]() -> std::pair<AttributedLoopType, std::string> {
        if (attrLoop.type == AttributedLoopType::Inner) {
            return {AttributedLoopType::Inner, "@inner"};
        }
        return {AttributedLoopType::Outer, "@outer"};
    }();
    auto height = loopInfo.getHeightSameType(metaLoopType);
    // Case of @outer @outer or @inner @inner
    if (isFirst && loopInfo.metadata.type[1] == metaLoopType) {
        ++height;
    }
    if (height > heightLimit) {
        return tl::make_unexpected(Error{
            {}, util::fmt("More than {} nested [{}] loops", heightLimit + 1, loopName).value()});
    }
    attrLoop.dim = static_cast<DimType>(height);
    return params;
}
}  // namespace

tl::expected<TileParams, Error> tileParamsHandleAutoDims(const TileParams& params,
                                                         OklLoopInfo& loopInfo,
                                                         size_t heightLimit) {
    TileParams res = params;
    return updateParamsDim(res, loopInfo, true, heightLimit).and_then([&](auto params) {
        return updateParamsDim(res, loopInfo, false, heightLimit);
    });
}
}  // namespace oklt
