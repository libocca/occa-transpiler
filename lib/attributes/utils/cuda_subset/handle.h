#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include <any>
#include "attributes/frontend/params/tile.h"
#include "core/transpiler_session/session_stage.h"

namespace oklt::cuda_subset {
// TODO: create aliasing for tl::expected<std::any, Error>
tl::expected<std::any, Error> handleTileAttribute(const clang::Attr*,
                                                  const clang::Stmt*,
                                                  const TileParams* params,
                                                  SessionStage&);
tl::expected<std::any, Error> handleInnerAttribute(const clang::Attr*,
                                                   const clang::Stmt*,
                                                   const AttributedLoop* params,
                                                   SessionStage&);
tl::expected<std::any, Error> handleOuterAttribute(const clang::Attr*,
                                                   const clang::Stmt*,
                                                   const AttributedLoop* params,
                                                   SessionStage&);
tl::expected<std::any, Error> handleAtomicAttribute(const clang::Attr*,
                                                    const clang::Stmt*,
                                                    SessionStage&);

tl::expected<std::any, Error> handleKernelAttribute(const clang::Attr*,
                                                    const clang::Decl*,
                                                    SessionStage&);
tl::expected<std::any, Error> handleSharedAttribute(const clang::Attr*,
                                                    const clang::Decl*,
                                                    SessionStage&);
tl::expected<std::any, Error> handleRestrictAttribute(const clang::Attr*,
                                                      const clang::Decl*,
                                                      SessionStage&);
}  // namespace oklt::cuda_subset
