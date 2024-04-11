#pragma once

#include "core/sema/okl_sema_ctx.h"
#include "oklt/core/error.h"

namespace clang {
class Stmt;
class AttributedStmt;
}  // namespace clang

namespace oklt {
tl::expected<void, Error> verifyLoops(OklSemaCtx::ParsedKernelInfo& kernelInfo);
const clang::AttributedStmt* getAttributedStmt(SessionStage& s, const clang::Stmt& stmt);
tl::expected<std::any, Error> handleChildAttr(SessionStage& s,
                                              const clang::Stmt& stmt,
                                              std::string_view name);
}  // namespace oklt
