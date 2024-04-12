#pragma oonce

#include <functional>
#include <vector>

#include <clang/Lex/Lexer.h>

namespace oklt {
std::vector<clang::Token> fetchTokens(
    clang::Preprocessor& pp,
    std::optional<std::function<bool(const clang::Token)>> = std::nullopt);
}
