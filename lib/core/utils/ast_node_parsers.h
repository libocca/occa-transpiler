#pragma once

#include <oklt/core/kernel_metadata.h>
#include "core/sema/okl_sema_info.h"

#include <tl/expected.hpp>

namespace clang {
struct ForStmt;
struct ASTContext;
struct Attr;
}  // namespace clang

namespace oklt {
struct Error;
class SessionStage;

tl::expected<OklLoopInfo, Error> parseForStmt(SessionStage& stage,
                                              const clang::ForStmt& s,
                                              const clang::Attr* a);
}  // namespace oklt
