#pragma once

#include <oklt/core/error.h>

#include "attributes/frontend/params/tile.h"
#include "core/sema/okl_sema_info.h"

#include <tl/expected.hpp>

namespace oklt {
tl::expected<TileParams, Error> tileParamsHandleAutoAxis(const TileParams& params,
                                                         OklLoopInfo& loopInfo,
                                                         size_t heightLimit = 2);
}
