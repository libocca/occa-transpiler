#include "core/utils/range_to_string.h"
#include <clang/AST/ASTContext.h>
#include <clang/AST/Expr.h>
#include <clang/Basic/SourceLocation.h>
#include <clang/Lex/Lexer.h>

namespace oklt {
using namespace clang;
std::string getSourceText(const clang::SourceRange& range, clang::ASTContext& ctx) {
    auto& sourceManager = ctx.getSourceManager();
    auto& opts = ctx.getLangOpts();
    return clang::Lexer::getSourceText(CharSourceRange::getCharRange(range), sourceManager, opts)
        .str();
}

std::string getSourceText(const clang::Expr& expr, clang::ASTContext& ctx) {
    auto& sourceManager = ctx.getSourceManager();
    auto& opts = ctx.getLangOpts();
    return clang::Lexer::getSourceText(
               CharSourceRange::getTokenRange(expr.getSourceRange()), sourceManager, opts)
        .str();
}

std::string prettyPrint(const clang::Stmt& stmt, clang::ASTContext& ctx) {
    std::string ret;
    llvm::raw_string_ostream os(ret);
    stmt.printPretty(os, nullptr, ctx.getPrintingPolicy());

    return ret;
}

}  // namespace oklt
