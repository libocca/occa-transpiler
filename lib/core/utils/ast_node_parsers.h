#include <oklt/core/kernel_metadata.h>

#include <tl/expected.hpp>
#include "core/sema/okl_sema_info.h"

#include <clang/AST/Attr.h>

namespace clang {
struct ForStmt;
struct ASTContext;
}  // namespace clang

namespace oklt {
struct Error;

tl::expected<OklLoopInfo, Error> parseForStmt(const clang::Attr& a,
                                               const clang::ForStmt& s,
                                               clang::ASTContext& ctx);
}  // namespace oklt
