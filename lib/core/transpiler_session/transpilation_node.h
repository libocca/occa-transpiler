#pragma once

#include <clang/AST/ASTTypeTraits.h>
#include "core/sema/okl_sema_ctx.h"

#include <deque>

namespace oklt {
struct TranspilationNode {
    OklSemaCtx::ParsedKernelInfo* ki;
    OklLoopInfo* li;
    const clang::Attr* attr;
    clang::DynTypedNode node;
};

using TranspilationNodes = std::deque<TranspilationNode>;

}  // namespace oklt
