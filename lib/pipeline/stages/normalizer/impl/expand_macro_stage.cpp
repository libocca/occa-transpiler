#include <oklt/core/error.h>

#include "core/transpiler_session/session_stage.h"

#include "pipeline/stages/normalizer/error_codes.h"
#include "pipeline/stages/normalizer/impl/expand_macro_stage.h"
#include "pipeline/stages/normalizer/impl/okl_attribute.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Analysis/MacroExpansionContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Tooling.h>

namespace {

using namespace clang;
using namespace oklt;

SourceLocation findPreviousTokenStart(SourceLocation Start,
                                      const SourceManager& SM,
                                      const LangOptions& LangOpts) {
    SourceLocation BeforeStart = Start.getLocWithOffset(-1);
    return Lexer::GetBeginningOfToken(BeforeStart, SM, LangOpts);
}

SourceLocation findPreviousTokenKind(SourceLocation Start,
                                     const SourceManager& SM,
                                     const LangOptions& LangOpts,
                                     tok::TokenKind TK) {
    while (true) {
        SourceLocation L = findPreviousTokenStart(Start, SM, LangOpts);

        Token T;
        if (Lexer::getRawToken(L, T, SM, LangOpts, /*IgnoreWhiteSpace=*/true))
            return SourceLocation();

        if (T.is(TK))
            return T.getLocation();

        Start = L;
    }
}

void expandAndInlineMacroWithOkl(Preprocessor& pp, SessionStage& stage) {
    auto ctx = std::make_unique<MacroExpansionContext>(pp.getLangOpts());
    ctx->registerForPreprocessor(pp);

    pp.EnterMainSourceFile();
    const auto& sm = pp.getSourceManager();
    auto& rewriter = stage.getRewriter();

    // parse all tokens firstly
    std::list<Token> macros;
    while (true) {
        Token tok{};
        pp.Lex(tok);

        if (tok.is(tok::eof)) {
            break;
        }

        if (tok.is(tok::unknown)) {
            // Check for '@' symbol
            auto spelling = pp.getSpelling(tok);
            if (spelling.empty() || spelling[0] != OKL_ATTR_MARKER) {
                break;
            }
            tok.setKind(tok::at);
        }

        // catch only valid macro loc
        auto loc = tok.getLocation();
        if (!loc.isValid() || !loc.isMacroID()) {
            continue;
        }

        // catch start of macro expension
        if (!Lexer::isAtStartOfMacroExpansion(loc, pp.getSourceManager(), pp.getLangOpts())) {
            continue;
        }

        macros.emplace_back(std::move(tok));
    }

    // do macro expansion
    std::set<StringRef> macroNames;
    for (const auto& tok : macros) {
        // and it's is in user file
        if (SrcMgr::isSystem(sm.getFileCharacteristic(tok.getLocation()))) {
            continue;
        }

        // then inline macro into token location
        auto expansionLoc = sm.getExpansionLoc(tok.getLocation());
        auto expanded = ctx->getExpandedText(expansionLoc);
        if (!expanded) {
            llvm::outs() << "no expanded macro under: " << expansionLoc.printToString(sm) << '\n';
            continue;
        }

        auto original = ctx->getOriginalText(expansionLoc);
        if (!original) {
            llvm::outs() << "no original macro under: " << expansionLoc.printToString(sm) << '\n';
            continue;
        }

#ifdef NORMALIZER_DEBUG_LOG
        llvm::outs() << "at " << expansionLoc.printToString(sm) << " inline macro "
                     << original.value() << " by " << expanded.value() << '\n';
#endif
        rewriter.ReplaceText(expansionLoc, original->size(), expanded.value());

        // in case of macro with args take only macro name
        macroNames.insert(original.value().split('(').first);
    }

    // get rid of macro hell
    for (const auto& name : macroNames) {
        auto* ii = pp.getIdentifierInfo(name);
        if (!ii) {
            continue;
        }

        auto md = pp.getMacroDefinition(ii);
        auto* mi = md.getMacroInfo();
        if (!mi) {
            continue;
        }

        // hash definitely there however clang pp doest not provide this info
        // so lets find it manually
        auto hashLoc = findPreviousTokenKind(
            mi->getDefinitionLoc(), sm, pp.getLangOpts(), tok::TokenKind::hash);
        if (hashLoc.isInvalid()) {
            // replace by 'identical' macro to not break anything
            auto noop = "void void\n";
            rewriter.ReplaceText({mi->getDefinitionLoc(), mi->getDefinitionEndLoc()}, noop);
        } else {
            // keep number of new lines
            auto lines = sm.getExpansionLineNumber(mi->getDefinitionEndLoc());
            lines -= sm.getExpansionLineNumber(mi->getDefinitionLoc());
            rewriter.ReplaceText({hashLoc, mi->getDefinitionEndLoc()}, std::string(lines, '\n'));
        }
    }

    pp.EndSourceFile();
}

struct MacroExpander : public clang::ASTFrontendAction {
    explicit MacroExpander(ExpandMacroStageInput& input, ExpandMacroStageOutput& output)
        : _input(input),
          _output(output),
          _session(*input.session) {
        (void)_input;
    }

   protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                          llvm::StringRef in_file) override {
        return nullptr;
    }

    bool BeginSourceFileAction(CompilerInstance& compiler) override {
        auto& pp = compiler.getPreprocessor();

        SessionStage stage{_session, compiler};
        expandAndInlineMacroWithOkl(pp, stage);

        _output.cppSrc = stage.getRewriterResultForMainFile();
        _output.cppIncs = stage.getRewriterResultForHeaders();

        return false;
    }

   private:
    ExpandMacroStageInput& _input;
    ExpandMacroStageOutput& _output;
    TranspilerSession& _session;
};
}  // namespace
namespace oklt {

ExpandMacroResult expandMacro(ExpandMacroStageInput input) {
    if (input.cppSrc.empty()) {
        llvm::outs() << "input source string is empty\n";
        auto error =
            makeError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING, "input source string is empty");
        return tl::make_unexpected(std::vector<Error>{error});
    }

#ifdef NORMALIZER_DEBUG_LOG
    llvm::outs() << "stage 0 OKL source:\n\n" << input.cppSrc << '\n';
#endif

    Twine tool_name = "okl-transpiler-normalization-to-gnu";
    Twine file_name("main_kernel.cpp");
    std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

    auto input_file = std::move(input.cppSrc);

    auto& sessionInput = input.session->input;
    for (const auto& define : sessionInput.defines) {
        std::string def = "-D" + define;
        args.push_back(std::move(def));
    }

    for (const auto& includePath : sessionInput.inlcudeDirectories) {
        std::string incPath = "-I" + includePath.string();
        args.push_back(std::move(incPath));
    }

    ExpandMacroStageOutput output = {.session = input.session};
    auto ok = tooling::runToolOnCodeWithArgs(
        std::make_unique<MacroExpander>(input, output), input_file, args, file_name, tool_name);

    if (!ok) {
        return tl::make_unexpected(std::move(output.session->getErrors()));
    }

#ifdef NORMALIZER_DEBUG_LOG
    llvm::outs() << "stage 1 inlined macros with OKL cpp source:\n\n" << output.cppSrc << '\n';
#endif

    return output;
}
}  // namespace oklt
