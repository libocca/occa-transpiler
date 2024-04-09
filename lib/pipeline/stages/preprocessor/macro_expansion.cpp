#include "core/lex/lexer.h"
#include "core/transpiler_session/session_stage.h"

#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_registry.h"
#include "pipeline/utils/okl_attribute_traverser.h"

#include <clang/Analysis/MacroExpansionContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/LiteralSupport.h>

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

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

size_t countRangeLines(const SourceRange& range, const SourceManager& sm) {
    auto lines = sm.getExpansionLineNumber(range.getEnd());
    lines -= sm.getExpansionLineNumber(range.getBegin());
    return lines;
}

void removeCommentAfterDirective(std::list<SourceRange>& comments,
                                 const SourceLocation& directiveLoc,
                                 const SourceManager& sm,
                                 oklt::Rewriter& rewriter) {
    FullSourceLoc directiveFullLoc(directiveLoc, sm);
    for (auto it = comments.begin(); it != comments.end(); ++it) {
        FullSourceLoc commentFullLoc(it->getBegin(), sm);

        if (commentFullLoc.getFileID() != directiveFullLoc.getFileID()) {
            continue;
        }

        auto commentLineNumber = commentFullLoc.getLineNumber();
        auto directiveLineNumber = directiveFullLoc.getLineNumber();

        if (commentLineNumber > directiveLineNumber) {
            break;
        }
        if (commentLineNumber != directiveLineNumber) {
            continue;
        }

        auto lines = countRangeLines(*it, sm);
        rewriter.ReplaceText(*it, std::string(lines, '\n'));
        comments.erase(it++);
    }
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
    bool isEnabled(SourceLocation loc, const MacroDefinition& md) {
        auto* mi = md.getMacroInfo();
        if (!mi) {
            return false;
        }

        return mi->isEnabled();
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
        if (sm.isInSystemHeader(loc)) {
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
                             std::string("if ") + (isEnabled(loc, md) ? "1" : "0"));
    }

    void Elifdef(SourceLocation loc,
                 const Token& macroNameTok,
                 const MacroDefinition& md) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }

        results.emplace_back(SourceRange{loc, macroNameTok.getLocation()},
                             std::string("elif ") + (isEnabled(loc, md) ? "1" : "0"));
    }

    void Elifdef(SourceLocation loc, SourceRange conditionRange, SourceLocation ifLoc) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }
        results.emplace_back(SourceRange{loc, conditionRange.getEnd()}, std::string("elif 0"));
    }

    void Ifndef(SourceLocation loc, const Token& macroNameTok, const MacroDefinition& md) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }

        auto* mi = md.getMacroInfo();
        if (!mi) {
            return;
        }
        if (mi->isUsedForHeaderGuard()) {
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
                             std::string("if ") + (isEnabled(loc, md) ? "0" : "1"));
    }

    void Elifndef(SourceLocation loc,
                  const Token& macroNameTok,
                  const MacroDefinition& md) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }

        results.emplace_back(SourceRange{loc, macroNameTok.getLocation()},
                             std::string("elif ") + (isEnabled(loc, md) ? "0" : "1"));
    }

    void Elifndef(SourceLocation loc, SourceRange conditionRange, SourceLocation IfLoc) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }
        results.emplace_back(SourceRange{loc, conditionRange.getEnd()}, std::string("elif 0"));
    }

    void Endif(SourceLocation loc, SourceLocation IfLoc) override {
        if (sm.isInSystemHeader(loc)) {
            return;
        }
        results.emplace_back(SourceRange{loc, loc.getLocWithOffset(5)}, std::string("endif"));
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
    std::vector<SourceRange> emptyMacros;
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

        emptyMacros.emplace_back(SourceRange{macroNameTok.getLocation(), range.getEnd()});
    }
};

struct CommentDeleter : public CommentHandler {
    std::list<SourceRange> comments;
    const SourceManager& sm;
    CommentDeleter(const SourceManager& sm)
        : sm(sm) {}

    bool HandleComment(Preprocessor&, SourceRange commentRange) override {
        if (sm.isInSystemHeader(commentRange.getBegin())) {
            return false;
        }
        comments.emplace_back(commentRange);
        return false;
    }
};

void preprocessMacros(SessionStage& stage, Preprocessor& pp) {
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

    auto commentDeleter = std::make_unique<CommentDeleter>(sm);
    pp.addCommentHandler(commentDeleter.get());

    // try to catch the first token of macro expansion
    auto& rewriter = stage.getRewriter();
    std::list<Token> macros;
    fetchTokens(pp, [&macros, &pp, &sm](const Token& tok) {
        // catch only valid macro loc
        auto loc = tok.getLocation();
        if (loc.isInvalid() || !loc.isMacroID()) {
            return true;
        }

        if (sm.isInSystemHeader(loc)) {
            return true;
        }

        // catch start of macro expansion
        if (!Lexer::isAtEndOfMacroExpansion(loc, sm, pp.getLangOpts())) {
            return true;
        }

        // catched start of the macro
        macros.emplace_back(std::move(tok));

        return true;
    });

    // TODO split into smaller processing function
    //
    //  do macro expansion
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
            SPDLOG_WARN("No expanded macro under: {}", expansionLoc.printToString(sm));
            continue;
        }

        auto original = ctx->getOriginalText(expansionLoc);
        if (!original) {
            SPDLOG_WARN("no original macro under: {}", expansionLoc.printToString(sm));
            continue;
        }

        // in case of macro with args take only macro name
        auto macroName = original->split('(').first;
        SPDLOG_DEBUG("Expansion at: {} inline macro {} by {}",
                     expansionLoc.printToString(sm),
                     original.value(),
                     expanded.value());
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
        auto lines = countRangeLines({mi->getDefinitionLoc(), mi->getDefinitionEndLoc()}, sm);
        rewriter.ReplaceText({hashLoc, mi->getDefinitionEndLoc()}, std::string(lines, '\n'));

        FullSourceLoc macroFullLoc(hashLoc, sm);
        removeCommentAfterDirective(commentDeleter->comments, macroFullLoc, sm, rewriter);
    }

    // remove empty macros from source code
    for (const auto& em : defCallback->emptyMacros) {
        rewriter.ReplaceText(em, "");
    }

    // macro can be under #if/#elif
    // in such case expansion ctx does not work so just replace macro dependent condition by
    // context free true/false
    for (const auto& c : condCallback->results) {
        if (c.conditionValue == PPCallbacks::CVK_NotEvaluated) {
            continue;
        }

        auto condExpanionRange = sm.getExpansionRange(c.conditionRange);
        rewriter.ReplaceText(condExpanionRange,
                             c.conditionValue == PPCallbacks::CVK_True ? "1" : "0");
        // remove comment behind the macro if such exists
        FullSourceLoc macroFullLoc(condExpanionRange.getBegin(), sm);
        removeCommentAfterDirective(commentDeleter->comments, macroFullLoc, sm, rewriter);
    }

    // macro can be under #ifdef/#elifdef etc
    // in such case expansion ctx does not work so just replace macro dependent condition by
    // context free true/false
    for (const auto& c : defCondCallback->results) {
        rewriter.ReplaceText(c.range, c.replacement);
        FullSourceLoc macroFullLoc(c.range.getBegin(), sm);
        removeCommentAfterDirective(commentDeleter->comments, macroFullLoc, sm, rewriter);
    }
}

class MacroExpansion : public StageAction {
   public:
    MacroExpansion() { _name = MACRO_EXPANSION_STAGE; };

    bool BeginSourceFileAction(clang::CompilerInstance& compiler) override {
        auto& pp = compiler.getPreprocessor();

        preprocessMacros(pp, *_stage);
        return true;
    }
};

StagePluginRegistry::Add<MacroExpansion> macroExpansion(MACRO_EXPANSION_STAGE, "");
}  // namespace
