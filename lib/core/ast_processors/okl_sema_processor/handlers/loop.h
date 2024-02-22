#pragma once

namespace clang {
struct ForStmt;
struct Attr;
}  // namespace clang

namespace oklt {
struct SessionStage;
struct OklSemaCtx;

// validator and translator for OKL attributed for loop
bool preValidateOklForLoopSema(const clang::Attr*,
                                const clang::ForStmt*,
                                SessionStage&,
                                OklSemaCtx& sema);
bool postValidateOklForLoopSema(const clang::Attr*,
                         const clang::ForStmt*,
                         SessionStage&,
                         OklSemaCtx& sema);

}  // namespace oklt
   // namespace oklt
