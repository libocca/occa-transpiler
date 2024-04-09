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
HandleResult preValidateOklForLoop(SessionStage&,
                                   OklSemaCtx& sema,
                                   const clang::ForStmt&,
                                   const clang::Attr&);

HandleResult preValidateOklForLoopWithoutAttribute(SessionStage&,
                                                   OklSemaCtx& sema,
                                                   const clang::ForStmt&,
                                                   const clang::Attr*);

HandleResult postValidateOklForLoop(SessionStage&,
                                    OklSemaCtx& sema,
                                    const clang::ForStmt&,
                                    const clang::Attr&);

HandleResult postValidateOklForLoopWithoutAttribute(SessionStage&,
                                                    OklSemaCtx& sema,
                                                    const clang::ForStmt&,
                                                    const clang::Attr*);

}  // namespace oklt
   // namespace oklt
