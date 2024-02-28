#include "attributes/frontend/params/tile.h"
#include "core/attribute_manager/result.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

namespace oklt::cuda_subset {
HandleResult handleTileAttribute(const clang::Attr*,
                                 const clang::ForStmt*,
                                 const TileParams* params,
                                 SessionStage&);
HandleResult handleInnerAttribute(const clang::Attr*,
                                  const clang::ForStmt*,
                                  const AttributedLoop* params,
                                  SessionStage&);
HandleResult handleOuterAttribute(const clang::Attr*,
                                  const clang::ForStmt*,
                                  const AttributedLoop* params,
                                  SessionStage&);
HandleResult handleAtomicAttribute(const clang::Attr*, const clang::Stmt*, SessionStage&);

HandleResult handleKernelAttribute(const clang::Attr*, const clang::Decl*, SessionStage&);
HandleResult handleSharedAttribute(const clang::Attr*, const clang::Decl*, SessionStage&);
HandleResult handleRestrictAttribute(const clang::Attr*, const clang::ParmVarDecl*, SessionStage&);
}  // namespace oklt::cuda_subset
