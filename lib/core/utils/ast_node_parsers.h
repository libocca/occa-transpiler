#include <oklt/core/kernel_metadata.h>

#include <tl/expected.hpp>
#include "core/sema/okl_sema_info.h"

namespace clang {
struct ForStmt;
struct ASTContext;
struct Attr;
}  // namespace clang

namespace oklt {
struct Error;
class SessionStage;

tl::expected<OklLoopInfo, Error> parseForStmt(const clang::Attr& a,
                                              const clang::ForStmt& s,
                                              SessionStage& stage);
}  // namespace oklt
