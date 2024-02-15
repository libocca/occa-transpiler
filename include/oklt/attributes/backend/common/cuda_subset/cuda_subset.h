#include <clang/AST/Decl.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt::cuda_subset {
bool handleGlobalConstant(const clang::Decl* decl, SessionStage& s);
bool handleGlobalFunction(const clang::Decl* decl, SessionStage& s);

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s);

}  // namespace oklt::cuda_subset
