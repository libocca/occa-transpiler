#include "attributes/attribute_names.h"

#include "core/lex/lexer.h"

#include <clang/Lex/Preprocessor.h>

namespace oklt {
using namespace clang;
std::vector<clang::Token> fetchTokens(
    clang::Preprocessor& pp,
    std::optional<std::function<bool(const clang::Token)>> watcher) {
    const auto& sm = pp.getSourceManager();
    std::vector<Token> tokens;

    pp.EnterMainSourceFile();
    while (true) {
        Token tok{};
        pp.Lex(tok);

        // only include tokens from user input source,headers
        if (sm.isInSystemHeader(tok.getLocation())) {
            continue;
        }

        if (tok.is(tok::eof)) {
            break;
        }

        if (tok.is(tok::unknown)) {
            // Check for '@' symbol
            auto spelling = pp.getSpelling(tok);
            if (spelling.empty() || spelling[0] != OKL_ATTR_NATIVE_MARKER) {
                break;
            }
            tok.setKind(tok::at);
        }

        tokens.push_back(tok);

        if (watcher) {
            (*watcher)(tok);
        }
    }
    pp.EndSourceFile();

    return tokens;
}
}  // namespace oklt
