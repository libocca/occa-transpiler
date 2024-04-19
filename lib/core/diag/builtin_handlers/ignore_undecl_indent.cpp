#include "core/diag/diag_handler.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <llvm/Support/ManagedStatic.h>

namespace {
using namespace oklt;
using namespace clang;

struct BraceCounter {
    unsigned short paren = 0;
    unsigned short bracket = 0;
    unsigned short brace = 0;
    explicit operator bool() const { return (paren != 0 || bracket != 0 || brace != 0); }
    void count(const Token& tok) {
        auto kind = tok.getKind();
        switch (kind) {
            case tok::l_paren:
                ++paren;
                break;
            case tok::r_paren:
                --paren;
                break;
            case tok::l_square:
                ++bracket;
                break;
            case tok::r_square:
                --bracket;
                break;
            case tok::l_brace:
                ++brace;
                break;
            case tok::r_brace:
                --brace;
                break;
            default:
                break;
        }
    }
};

bool isAllowedTok(tok::TokenKind K) {
    switch (K) {
        case tok::l_paren:
        case tok::r_paren:
        case tok::l_square:
        case tok::r_square:
        case tok::l_brace:
        case tok::r_brace:
        case tok::identifier:
        case tok::raw_identifier:
        case tok::coloncolon:
            return true;
        default:
            break;
    }

    return false;
}

bool isLooksLikeFunctionCall(const SourceLocation& loc,
                             const SourceManager& sm,
                             const LangOptions& lo) {
    bool hasLParen = false;
    bool hasRParen = false;

    std::optional<Token> token = Token{};
    token->setKind(tok::identifier);

    BraceCounter cnt;
    for (SourceLocation currLoc = loc; cnt || isAllowedTok(token->getKind());
         currLoc = token->getLocation()) {
        token = Lexer::findNextToken(currLoc, sm, lo);
        if (!token || token->is(tok::unknown)) {
            return false;
        }

        if (!hasLParen && token->is(tok::r_paren)) {
            return false;
        }

        if (!cnt && token->is(tok::l_paren) && !hasLParen) {
            hasLParen = true;
        }

        cnt.count(token.value());
        if (cnt) {
            continue;
        }

        if (token->is(tok::r_paren)) {
            hasRParen = true;
            break;
        }
    }

    return (!cnt && hasLParen && hasRParen);
}

class IgnoreUndeclHandler : public DiagHandler {
   public:
    IgnoreUndeclHandler(unsigned id)
        : DiagHandler(id){};
    IgnoreUndeclHandler()
        : IgnoreUndeclHandler(diag::err_undeclared_var_use){};

    bool HandleDiagnostic(SessionStage& session, DiagLevel level, const Diagnostic& info) override {
        const auto& sm = info.getSourceManager();

        // XXX In general OKL allows malformed source code to be parsed successfully
        // For example undeclared variable/function calls are suppressed and let target back-end
        // compiler figure out
        // this design is questionable but anyway so far we obey legacy
        if (!isLooksLikeFunctionCall(info.getLocation(), sm, session.getCompiler().getLangOpts())) {
            return false;
        }
        // TODO unify with error reporting to get correct location and source line
        llvm::SmallString<64> buf;
        info.FormatDiagnostic(buf);
        std::string msg{buf.begin(), buf.end()};

        FullSourceLoc loc(info.getLocation(), sm);
        session.pushWarning(std::to_string(loc.getLineNumber()) + ":" + msg);

        return true;
    }
};

class IgnoreUndeclSuggestHandler : public IgnoreUndeclHandler {
   public:
    IgnoreUndeclSuggestHandler()
        : IgnoreUndeclHandler(diag::err_undeclared_var_use_suggest){};
};

oklt::DiagHandlerRegistry::Add<IgnoreUndeclHandler> diag_dim("IgnoreUndeclUse", "");
oklt::DiagHandlerRegistry::Add<IgnoreUndeclSuggestHandler> diag_dim2("IgnoreUndeclSuggestUse", "");
}  // namespace
