#include <oklt/core/kernel_metadata.h>
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/Attr.h>
#include <clang/AST/Stmt.h>

#include <string>

namespace oklt {
std::string getCondCompStr(const BinOp& bo);
std::string getUnaryStr(const UnOp& uo, const std::string& var);
std::string buildCloseScopes(int& openedScopeCounter);
void replaceAttributedLoop(const clang::Attr* a,
                           const clang::ForStmt* f,
                           const std::string& prefixCode,
                           const std::string& suffixCode,
                           SessionStage& s);
}  // namespace oklt
