#include "core/attribute_manager/attribute_manager.h"
#include "core/sema/okl_sema_ctx.h"
#include "core/transpilation.h"

namespace oklt::openmp {
HandleResult postHandleExclusive(OklLoopInfo& loopInfo, TranspilationBuilder& trans);
HandleResult postHandleShared(OklLoopInfo& loopInfo, TranspilationBuilder& trans);
}  // namespace oklt::openmp
