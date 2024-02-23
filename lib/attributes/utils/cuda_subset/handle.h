#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include "core/transpiler_session/session_stage.h"

namespace oklt::cuda_subset {
bool handleTileAttribute(const clang::Attr*, const clang::Stmt*, SessionStage&);
bool handleInnerAttribute(const clang::Attr*, const clang::Stmt*, SessionStage&);
bool handleOuterAttribute(const clang::Attr*, const clang::Stmt*, SessionStage&);
bool handleAtomicAttribute(const clang::Attr*, const clang::Stmt*, SessionStage&);

bool handleKernelAttribute(const clang::Attr*, const clang::Decl*, SessionStage&);
bool handleSharedAttribute(const clang::Attr*, const clang::Decl*, SessionStage&);
bool handleRestrictAttribute(const clang::Attr*, const clang::Decl*, SessionStage&);
}  // namespace oklt::cuda_subset
