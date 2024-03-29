#pragma once

#include "core/attribute_manager/result.h"

namespace clang {
struct ForStmt;
struct Attr;
}  // namespace clang

namespace oklt {
class SessionStage;
class OklSemaCtx;

// validator and translator for OKL attributed for loop
HandleResult preValidateOklForLoop(const clang::Attr&,
                                   const clang::ForStmt&,
                                   OklSemaCtx& sema,
                                   SessionStage&);

HandleResult preValidateOklForLoopWithoutAttribute(const clang::Attr*,
                                                   const clang::ForStmt&,
                                                   OklSemaCtx& sema,
                                                   SessionStage&);

HandleResult postValidateOklForLoop(const clang::Attr&,
                                    const clang::ForStmt&,
                                    OklSemaCtx& sema,
                                    SessionStage&);

HandleResult postValidateOklForLoopWithoutAttribute(const clang::Attr*,
                                                    const clang::ForStmt&,
                                                    OklSemaCtx& sema,
                                                    SessionStage&);

}  // namespace oklt
   // namespace oklt
