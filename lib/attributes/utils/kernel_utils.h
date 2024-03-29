#pragma once

#include "core/sema/okl_sema_ctx.h"
#include "oklt/core/error.h"

namespace clang {
class Stmt;
class AttributedStmt;
}  // namespace clang

namespace oklt {
    tl::expected<void, Error> verifyLoops(OklSemaCtx::ParsedKernelInfo& kernelInfo);
const clang::AttributedStmt* getAttributedStmt(const clang::Stmt& stmt, SessionStage& s);
tl::expected<void, Error> handleChildAttr(const clang::Stmt& stmt,
                                          std::string_view name,
                                          SessionStage& s);
}  // namespace oklt
