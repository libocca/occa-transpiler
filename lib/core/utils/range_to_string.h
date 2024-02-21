#pragma once

#include <string>

namespace clang {
class ASTContext;
class SourceRange;
class Expr;
}  // namespace clang

namespace oklt {
std::string getSourceText(const clang::SourceRange& range, clang::ASTContext& ctx);
std::string getSourceText(const clang::Expr& expr, clang::ASTContext& ctx);
}  // namespace oklt
