#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>
#include "core/transpiler_session/session_stage.h"

namespace oklt::cuda_subset {
bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s);
bool handleKernelAttribute(const clang::Attr* a, const clang::Decl* d, SessionStage& s);
}  // namespace oklt