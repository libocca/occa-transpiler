#pragma once

#include "core/sema/okl_sema_ctx.h"
#include "oklt/core/error.h"

namespace oklt {
    tl::expected<void, Error> verifyLoops(OklSemaCtx::ParsedKernelInfo& kernelInfo);
}