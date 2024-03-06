#pragma once

#include "core/attribute_manager/result.h"

namespace clang {
struct DeclRefExpr;
}  // namespace clang

namespace oklt {
struct SessionStage;
struct OklSemaCtx;

// validator and translator for OKL attributed for loop
HandleResult preValidateDeclRefExpr(const clang::DeclRefExpr&, OklSemaCtx& sema, SessionStage&);
HandleResult postValidateDeclRefExpr(const clang::DeclRefExpr&, OklSemaCtx& sema, SessionStage&);

}  // namespace oklt
   // namespace oklt
