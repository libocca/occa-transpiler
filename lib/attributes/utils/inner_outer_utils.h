#include "attributes/frontend/params/loop.h"
#include "core/sema/okl_sema_info.h"
#include "oklt/core/error.h"
#include "oklt/core/kernel_metadata.h"

#include <tl/expected.hpp>

namespace oklt {
tl::expected<AttributedLoop, Error> innerOuterParamsHandleAutoDims(
    const AttributedLoop& params,
    OklLoopInfo& loopInfo,
    const AttributedLoopType& loopType,
    size_t heightLimit = 2);
}  // namespace oklt
