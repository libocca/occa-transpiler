#include <oklt/core/kernel_metadata.h>
#include <string>
// #include <clang/AST/Expr.h>

namespace oklt {
std::string getCondCompStr(const BinOp& bo);
std::string getUnaryStr(const UnOp& uo, const std::string& var);
std::string buildCloseScopes(int& openedScopeCounter);
}  // namespace oklt
