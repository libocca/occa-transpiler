#include <string>
#include <clang/AST/Expr.h>

namespace oklt {
std::string getCondCompStr(const clang::BinaryOperator::Opcode& bo);
std::string getUnaryStr(const clang::UnaryOperator::Opcode& uo, const std::string& var);
std::string buildCloseScopes(int& openedScopeCounter);
}
