#pragma once

#include <string>
#include "clang/AST/Stmt.h"

#include <clang/Rewrite/Core/Rewriter.h>

namespace clang {
class ASTContext;
class SourceRange;
class Rewriter;
class Expr;
}  // namespace clang

namespace oklt {
std::string getSourceText(const clang::SourceRange& range, clang::ASTContext& ctx);
std::string getSourceText(const clang::Expr& expr, clang::ASTContext& ctx);
std::string getSourceText(const clang::Stmt& stmt, clang::ASTContext& ctx);

template<typename NodeType>
std::string getLatestSourceText(const NodeType& node, clang::Rewriter& rewriter) {
    // TODO: doesn't work correctly if node is macros
    return rewriter.getRewrittenText(node.getSourceRange());
}

template<typename NodeType>
std::string getLatestSourceText(const NodeType* node, clang::Rewriter& rewriter) {
    // TODO: doesn't work correctly if node is macros
    return rewriter.getRewrittenText(node->getSourceRange());
}
}  // namespace oklt
