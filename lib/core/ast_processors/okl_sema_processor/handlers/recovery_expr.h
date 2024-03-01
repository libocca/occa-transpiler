#pragma once

#include "core/attribute_manager/result.h"

namespace clang {
struct RecoveryExpr;
}  // namespace clang

namespace oklt {
struct SessionStage;
struct OklSemaCtx;

// validator and translator for OKL attributed for loop
HandleResult preValidateRecoveryExpr(const clang::RecoveryExpr&, OklSemaCtx& sema, SessionStage&);
HandleResult postValidateRecoveryExpr(const clang::RecoveryExpr&, OklSemaCtx& sema, SessionStage&);

}  // namespace oklt
   // namespace oklt
