#pragma once

#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"

#include <clang/Rewrite/Core/Rewriter.h>

namespace oklt::openmp {
HandleResult postHandleExclusive(OklLoopInfo& loopInfo, clang::Rewriter& rewriter);
HandleResult postHandleShared(OklLoopInfo& loopInfo, clang::Rewriter& rewriter);
}  // namespace oklt::openmp
