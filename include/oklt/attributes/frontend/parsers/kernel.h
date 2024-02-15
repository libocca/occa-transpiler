#include <clang/AST/Attr.h>
#include <oklt/core/transpiler_session/session_stage.h>

namespace oklt {
bool parseKernelAttribute(const clang::Attr* a, SessionStage& s);
}