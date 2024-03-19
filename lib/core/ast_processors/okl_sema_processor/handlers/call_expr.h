#pragma once

#include "core/attribute_manager/result.h"

namespace clang {
struct CallExpr;
struct Attr;
}  // namespace clang

namespace oklt {
class SessionStage;
class OklSemaCtx;

// validator and translator for OKL attributed for loop
HandleResult preValidateCallExpr(const clang::Attr*,
                                 const clang::CallExpr&,
                                 OklSemaCtx&,
                                 SessionStage&);
HandleResult postValidateCallExpr(const clang::Attr*,
                                  const clang::CallExpr&,
                                  OklSemaCtx&,
                                  SessionStage&);

}  // namespace oklt
   // namespace oklt
