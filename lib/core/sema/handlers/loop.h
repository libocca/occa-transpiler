#pragma once

#include "core/handler_manager/result.h"

namespace clang {
struct ForStmt;
struct Attr;
}  // namespace clang

namespace oklt {
class SessionStage;
class OklSemaCtx;

// validator and translator for OKL attributed for loop
HandleResult preValidateOklForLoop(SessionStage&,
                                   const clang::ForStmt&,
                                   const clang::Attr&);

HandleResult preValidateOklForLoopWithoutAttribute(SessionStage&,
                                                   const clang::ForStmt&,
                                                   const clang::Attr*);

HandleResult postValidateOklForLoop(SessionStage&,
                                    const clang::ForStmt&,
                                    const clang::Attr&);

HandleResult postValidateOklForLoopWithoutAttribute(SessionStage&,
                                                    const clang::ForStmt&,
                                                    const clang::Attr*);

}  // namespace oklt
   // namespace oklt
