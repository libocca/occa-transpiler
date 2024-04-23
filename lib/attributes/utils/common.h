#pragma once

#include "core/sema/okl_sema_info.h"

namespace oklt {
// Used for @shated and @exclusive, since we can define them only between @outer and @inner loops
bool isLastOuter(OklLoopInfo* loop);

}  // namespace oklt
