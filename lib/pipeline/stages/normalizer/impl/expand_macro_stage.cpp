#include <oklt/core/error.h>

#include "core/transpiler_session/session_stage.h"
#include "core/vfs/overlay_fs.h"

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

SourceLocation findPreviousTokenStart(SourceLocation start,
                                      const SourceManager& sm,
                                      const LangOptions& langOpts) {
    SourceLocation beforeStart = start.getLocWithOffset(-1);
    return Lexer::GetBeginningOfToken(beforeStart, sm, langOpts);
}

SourceLocation findPreviousTokenKind(SourceLocation start,
                                     const SourceManager& sm,
                                     const LangOptions& langOpts,
                                     tok::TokenKind tk) {
    while (true) {
        SourceLocation l = findPreviousTokenStart(start, sm, langOpts);

        Token t;
        if (Lexer::getRawToken(l, t, sm, langOpts, /*IgnoreWhiteSpace=*/true))
            return SourceLocation();

        if (t.is(tk))
            return t.getLocation();

        start = l;
    }
}

bool isHeaderGuardEx(StringRef name) {
    // XXX last drastic check for header guard
    //  header guard could be not the first and last line of source and clang fails to catch
    //  this lets try simple check my naming convention
    //  TODO add more complex check by catch ifndef/endif
    if (name.ends_with("_H") || name.ends_with("_H_") || name.ends_with("_HPP") ||
        name.ends_with("_HPP_")) {
        return true;
    }

    return false;
}

class CondDirectiveCallbacks : public PPCallbacks {
   public:
    struct Result {
        SourceRange conditionRange;
        ConditionValueKind conditionValue;

        Result(SourceRange r, ConditionValueKind k)
            : conditionRange(r),
              conditionValue(k) {}
    };

    std::vector<Result> results;
    CondDirectiveCallbacks(const SourceManager& sm_)
        : sm(sm_) {}
    const SourceManager& sm;

    void If(SourceLocation loc,
            SourceRange conditionRange,
            ConditionValueKind conditionValue) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }
        results.emplace_back(conditionRange, conditionValue);
    }

    void Elif(SourceLocation loc,
              SourceRange conditionRange,
              ConditionValueKind conditionValue,
              SourceLocation ifLoc) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }
        results.emplace_back(conditionRange, conditionValue);
    }
};

class DefCondDirectiveCallbacks : public PPCallbacks {
   private:
    MacroInfo* getMacroInfoForUserMacro(SourceLocation loc, const MacroDefinition& md) {
        if (sm.isInSystemHeader(loc)) {
            return nullptr;
        }
        auto* mi = md.getMacroInfo();
        if (!mi) {
            return nullptr;
        }
        if (mi->isUsedForHeaderGuard()) {
            return nullptr;
        }

        return mi;
    }

   public:
    struct Result {
        SourceRange range;
        std::string replacement;

        Result(SourceRange range_, std::string replacement_)
            : range(range_),
              replacement(replacement_) {}
    };

    std::vector<Result> results;
    DefCondDirectiveCallbacks(const SourceManager& sm_)
        : sm(sm_) {}
    const SourceManager& sm;

    void Ifdef(SourceLocation loc, const Token& macroNameTok, const MacroDefinition& md) override {
        auto mi = getMacroInfoForUserMacro(loc, md);
        if (!mi) {
            return;
        }

        auto* id = macroNameTok.getIdentifierInfo();
        if (!id) {
            return;
        }
        if (isHeaderGuardEx(id->getName())) {
            return;
        }

        results.emplace_back(SourceRange{loc, macroNameTok.getLocation()},
                             std::string("if ") + (mi->isEnabled() ? "1" : "0"));
    }

    void Elifdef(SourceLocation loc,
                 const Token& macroNameTok,
                 const MacroDefinition& md) override {
        auto mi = getMacroInfoForUserMacro(loc, md);
        if (!mi) {
            return;
        }
        results.emplace_back(SourceRange{loc, macroNameTok.getLocation()},
                             std::string("elif ") + (mi->isEnabled() ? "1" : "0"));
    }

    void Elifdef(SourceLocation loc, SourceRange conditionRange, SourceLocation ifLoc) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }
        results.emplace_back(SourceRange{loc, conditionRange.getEnd()}, std::string("elif 0"));
    }

    void Ifndef(SourceLocation loc, const Token& macroNameTok, const MacroDefinition& md) override {
        auto mi = getMacroInfoForUserMacro(loc, md);
        if (!mi) {
            return;
        }

        auto* id = macroNameTok.getIdentifierInfo();
        if (!id) {
            return;
        }
        if (isHeaderGuardEx(id->getName())) {
            return;
        }

        results.emplace_back(SourceRange{loc, macroNameTok.getLocation()},
                             std::string("if ") + (mi->isEnabled() ? "0" : "1"));
    }

    void Elifndef(SourceLocation loc,
                  const Token& macroNameTok,
                  const MacroDefinition& md) override {
        auto mi = getMacroInfoForUserMacro(loc, md);
        if (!mi) {
            return;
        }
        results.emplace_back(SourceRange{loc, macroNameTok.getLocation()},
                             std::string("elif ") + (mi->isEnabled() ? "0" : "1"));
    }

    void Elifndef(SourceLocation loc, SourceRange conditionRange, SourceLocation IfLoc) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }
        results.emplace_back(SourceRange{loc, conditionRange.getEnd()}, std::string("elif 0"));
    }
};

class DefineDirectiveCallbacks : public PPCallbacks {
   private:
   public:
    struct Result {
        std::string name;
        const MacroDirective* md;
        Result(const StringRef name_, const MacroDirective* md_)
            : name(name_),
              md(md_) {}
    };

    std::vector<Result> results;
    std::vector<Token> emptyMacros;
    DefineDirectiveCallbacks(const SourceManager& sm_)
        : sm(sm_) {}
    const SourceManager& sm;

    void MacroDefined(const Token& macroNameTok, const MacroDirective* md) override {
        if (sm.isInSystemHeader(macroNameTok.getLocation())) {
            return;
        }

        const auto* id = macroNameTok.getIdentifierInfo();
        if (!id) {
            return;
        }

        auto* mi = md->getMacroInfo();
        if (!mi) {
            return;
        }

        if (mi->isUsedForHeaderGuard()) {
            return;
        }

        if (isHeaderGuardEx(id->getName())) {
            return;
        }

        results.emplace_back(id->getName(), md);
    }

    void MacroExpands(const Token& macroNameTok,
                      const MacroDefinition& md,
                      SourceRange range,
                      const MacroArgs* args) override {
        if (sm.isInSystemHeader(macroNameTok.getLocation())) {
            return;
        }

        const auto* id = macroNameTok.getIdentifierInfo();
        if (!id) {
            return;
        }

        auto* mi = md.getMacroInfo();
        if (!mi) {
            return;
        }

        if (mi->isUsedForHeaderGuard()) {
            return;
        }

        if (isHeaderGuardEx(id->getName())) {
            return;
        }

        if (mi->getNumTokens() != 0) {
            return;
        }

        emptyMacros.emplace_back(macroNameTok);
    }
};

std::list<Token> lexMacroToken(Preprocessor& pp) {
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
        if (loc.isInvalid() || !loc.isMacroID()) {
            continue;
        }

        if (pp.getSourceManager().isInSystemHeader(loc)) {
            continue;
        }

        // catch start of macro expansion
        if (!Lexer::isAtEndOfMacroExpansion(loc, pp.getSourceManager(), pp.getLangOpts())) {
            continue;
        }

        macros.emplace_back(std::move(tok));
    }

    return macros;
}

void expandAndInlineMacroWithOkl(Preprocessor& pp, SessionStage& stage) {
    auto ctx = std::make_unique<MacroExpansionContext>(pp.getLangOpts());
    ctx->registerForPreprocessor(pp);

    const auto& sm = pp.getSourceManager();

    // intercept all conditions to replace them by non context static value
    auto* condCallback = new CondDirectiveCallbacks(sm);
    pp.addPPCallbacks(std::unique_ptr<PPCallbacks>(condCallback));

    // intercept all def conditions to replace them by non context static value
    auto* defCondCallback = new DefCondDirectiveCallbacks(sm);
    pp.addPPCallbacks(std::unique_ptr<PPCallbacks>(defCondCallback));

    // intercept all macro definition in case of multiple definitions to hace source info about each
    auto* defCallback = new DefineDirectiveCallbacks(sm);
    pp.addPPCallbacks(std::unique_ptr<PPCallbacks>(defCallback));

    pp.EnterMainSourceFile();
    auto& rewriter = stage.getRewriter();

    auto macros = lexMacroToken(pp);

    // do macro expansion
    std::set<StringRef> macroNames;
    for (const auto& tok : macros) {
        if (tok.hasLeadingEmptyMacro()) {
            rewriter.RemoveText({tok.getLocation().getLocWithOffset(-1), tok.getLocation()});
            continue;
        }

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

        // in case of macro with args take only macro name
        auto macroName = original->split('(').first;
#ifdef NORMALIZER_DEBUG_LOG
        llvm::outs() << "expansion at: " << expansionLoc.printToString(sm) << " inline macro "
                     << original.value() << " by " << expanded.value() << '\n';
#endif
        // expand macro in source code
        rewriter.ReplaceText(expansionLoc, original->size(), expanded.value());

        // save macro name to delete its definition later
        macroNames.insert(macroName);
    }

    // get rid of macro hell - try to remove all users macro except header guards
    for (const auto& defined : defCallback->results) {
        if (!defined.md) {
            continue;
        }
        auto* mi = defined.md->getMacroInfo();
        if (!mi) {
            continue;
        }

        if (mi->isUsedForHeaderGuard()) {
            continue;
        }

        auto hashLoc = findPreviousTokenKind(
            mi->getDefinitionLoc(), sm, pp.getLangOpts(), tok::TokenKind::hash);
        auto lines = sm.getExpansionLineNumber(mi->getDefinitionEndLoc());
        lines -= sm.getExpansionLineNumber(mi->getDefinitionLoc());
        rewriter.ReplaceText({hashLoc, mi->getDefinitionEndLoc()}, std::string(lines, '\n'));
    }

    // remove empty macros from source code
    for (const auto& em : defCallback->emptyMacros) {
        rewriter.RemoveText({em.getLocation(), em.getLocation()});
    }

    // macro can be under #if/#elif
    // in such case expansion ctx does not work so just replace macro dependent condition by
    // context free true/false
    for (auto& c : condCallback->results) {
        if (c.conditionValue == PPCallbacks::CVK_NotEvaluated) {
            continue;
        }

        rewriter.ReplaceText(sm.getExpansionRange(c.conditionRange),
                             c.conditionValue == PPCallbacks::CVK_True ? "1" : "0");
    }

    // macro can be under #ifdef/#elifdef etc
    // in such case expansion ctx does not work so just replace macro dependent condition by
    // context free true/false
    for (auto& c : defCondCallback->results) {
        rewriter.ReplaceText(c.range, c.replacement);
    }
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

    bool PrepareToExecuteAction(CompilerInstance& compiler) override {
        if (compiler.hasFileManager()) {
            auto overlayFs =
                makeOverlayFs(compiler.getFileManager().getVirtualFileSystemPtr(), _input.cppIncs);
            compiler.getFileManager().setVirtualFileSystem(overlayFs);
        }

        return true;
    }

    bool BeginSourceFileAction(CompilerInstance& compiler) override {
        auto& pp = compiler.getPreprocessor();

        SessionStage stage{_session, compiler};
        expandAndInlineMacroWithOkl(pp, stage);

        _output.cppSrc = stage.getRewriterResultForMainFile();
        _output.cppIncs = stage.getRewriterResultForHeaders();
        _output.cppIncs.fileMap.merge(_input.cppIncs.fileMap);

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
