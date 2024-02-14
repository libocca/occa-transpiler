#include <clang/AST/Decl.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt::cuda_like {
bool handleGlobalConstant(const clang::Decl* decl, SessionStage& s);
bool handleGlobalFunction(const clang::Decl* decl, SessionStage& s);
}  // namespace oklt::cuda_like
