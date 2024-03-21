#include "core/diag/diag_handler.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <llvm/Support/ManagedStatic.h>

namespace {
using namespace oklt;
using namespace clang;

bool isLooksLikeFunctionCall(const SourceLocation& loc,
                             const SourceManager& sm,
                             const LangOptions& lo) {
    std::list<Token> tokens;
    SourceLocation currLoc = loc;
    while (1) {
        auto maybeToken = Lexer::findNextToken(currLoc, sm, lo);
        if (!maybeToken) {
            return false;
        }

        if (maybeToken->is(tok::semi)) {
            break;
        }

        if (maybeToken->isOneOf(tok::l_paren, tok::r_paren)) {
            tokens.push_back(maybeToken.value());
        }

        currLoc = maybeToken->getLocation();
    }

    // at least two token for function call 'func()'
    if (tokens.size() < 2) {
        return false;
    }

    return tokens.front().is(tok::l_paren) && tokens.back().is(tok::r_paren);
}

class IgnoreUndeclHandler : public DiagHandler {
   public:
    IgnoreUndeclHandler()
        : DiagHandler(diag::err_undeclared_var_use){};

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
        session.pushWarning(std::to_string(loc.getLineNumber()) + ":" + msg.data());

        return true;
    }
};

oklt::DiagHandlerRegistry::Add<IgnoreUndeclHandler> diag_dim("IgnoreUndeclUse", "");
}  // namespace
