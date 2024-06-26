#pragma once

#include "core/rewriter/rewriter_proxy.h"

#include <clang/Lex/Lexer.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include <string>

namespace clang {
class ASTContext;
class Expr;
class Stmt;
class Decl;
}  // namespace clang

namespace oklt {
std::string getSourceText(const clang::SourceRange& range, clang::ASTContext& ctx);
std::string getSourceText(const clang::Expr& expr, clang::ASTContext& ctx);

template <typename NodeType>
std::string getSourceText(const NodeType& node, const oklt::Rewriter& rewriter) {
    auto& sourceManager = rewriter.getSourceMgr();
    auto& opts = rewriter.getLangOpts();
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(node.getSourceRange()), sourceManager, opts)
        .str();
}

template <typename NodeType>
std::string getLatestSourceText(const NodeType& node, const oklt::Rewriter& rewriter) {
    return rewriter.getRewrittenText(node.getSourceRange());
}

template <typename NodeType>
std::string getLatestSourceText(const NodeType* node, const oklt::Rewriter& rewriter) {
    return getLatestSourceText(*node, rewriter);
}
}  // namespace oklt
