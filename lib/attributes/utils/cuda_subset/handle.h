#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include <any>
#include "core/transpiler_session/session_stage.h"

namespace oklt::cuda_subset {
tl::expected<std::any, Error> handleTileAttribute(const clang::Attr*,
                                                  const clang::Stmt*,
                                                  const std::any& params,
                                                  SessionStage&);
tl::expected<std::any, Error> handleInnerAttribute(const clang::Attr*,
                                                   const clang::Stmt*,
                                                   const std::any& params,
                                                   SessionStage&);
tl::expected<std::any, Error> handleOuterAttribute(const clang::Attr*,
                                                   const clang::Stmt*,
                                                   const std::any& params,
                                                   SessionStage&);
tl::expected<std::any, Error> handleAtomicAttribute(const clang::Attr*,
                                                    const clang::Stmt*,
                                                    const std::any& params,
                                                    SessionStage&);

tl::expected<std::any, Error> handleKernelAttribute(const clang::Attr*,
                                                    const clang::Decl*,
                                                    const std::any& params,
                                                    SessionStage&);
tl::expected<std::any, Error> handleSharedAttribute(const clang::Attr*,
                                                    const clang::Decl*,
                                                    const std::any& params,
                                                    SessionStage&);
tl::expected<std::any, Error> handleRestrictAttribute(const clang::Attr*,
                                                      const clang::Decl*,
                                                      const std::any& params,
                                                      SessionStage&);
}  // namespace oklt::cuda_subset
