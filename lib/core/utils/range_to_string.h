#pragma once

#include <clang/AST/Expr.h>
#include <clang/AST/Stmt.h>
#include <clang/AST/Type.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Lex/Lexer.h>
#include <clang/Rewrite/Core/Rewriter.h>

#include <string>

namespace clang {
class ASTContext;
class SourceRange;
class Rewriter;
class Expr;
}  // namespace clang

// TODO: this is temporary thing until bug with rewriter is found out
constexpr bool IGNORE_REWRITTEN_TEXT = false;

namespace oklt {
std::string getSourceText(const clang::SourceRange& range, clang::ASTContext& ctx);
std::string getSourceText(const clang::Expr& expr, clang::ASTContext& ctx);
std::string getSourceText(const clang::Stmt& stmt, clang::ASTContext& ctx);

template <typename NodeType>
std::string getSourceText(const NodeType& node, const clang::Rewriter& rewriter) {
    auto& sourceManager = rewriter.getSourceMgr();
    auto& opts = rewriter.getLangOpts();
    return clang::Lexer::getSourceText(
               clang::CharSourceRange::getTokenRange(node.getSourceRange()), sourceManager, opts)
        .str();
}

// TODO: context is added temporary
template <typename NodeType>
std::string getLatestSourceText(const NodeType& node,
                                const clang::Rewriter& rewriter) {
    if constexpr (IGNORE_REWRITTEN_TEXT) {
        return getSourceText(node, rewriter);
    } else {
        // auto range = clang::CharSourceRange::getCharRange(node.getSourceRange());
        auto range = clang::CharSourceRange::getTokenRange(node.getSourceRange());
        return rewriter.getRewrittenText(range);
    }
}

template <typename NodeType>
std::string getLatestSourceText(const NodeType* node,
                                const clang::Rewriter& rewriter) {
    return getLatestSourceText(*node, rewriter);
}
}  // namespace oklt
