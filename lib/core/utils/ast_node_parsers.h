#include <oklt/core/kernel_metadata.h>

#include <tl/expected.hpp>

namespace clang {
struct ForStmt;
struct ASTContext;
}  // namespace clang

namespace oklt {
struct Error;
struct LoopMetaData;

tl::expected<LoopMetaData, Error> parseForStmt(const clang::ForStmt& s, clang::ASTContext& ctx);
}  // namespace oklt
