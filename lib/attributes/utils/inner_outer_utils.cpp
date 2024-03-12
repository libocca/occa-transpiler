#include <oklt/util/string_utils.h>

#include "attributes/utils/inner_outer_utils.h"

namespace oklt {
tl::expected<AttributedLoop, Error> innerOuterParamsHandleAutoDims(
    const AttributedLoop& params,
    OklLoopInfo& loopInfo,
    const LoopType& loopType,
    size_t heightLimit) {
    AttributedLoop res = params;
    if (res.dim == DimType::Auto) {
        auto height = loopInfo.getHeightSameType(loopType);
        if (height > heightLimit) {
            return tl::make_unexpected(Error{
                {}, util::fmt("More than {} nested [@inner] loops", heightLimit + 1).value()});
        }
        res.dim = static_cast<DimType>(height);
    }
    return res;
}
}  // namespace oklt
