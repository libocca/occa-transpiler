#include "core/lex/lexer.h"
#include "core/transpiler_session/session_stage.h"

#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_registry.h"
#include "pipeline/stages/normalizer/error_codes.h"
#include "pipeline/utils/okl_attribute.h"
#include "pipeline/utils/okl_attribute_traverser.h"

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/LiteralSupport.h>

#include <spdlog/spdlog.h>

namespace {
using namespace oklt;
using namespace clang;

std::set<std::string> macroAttrs{"directive"};

bool isProbablyMacroAttr(const OklAttribute& attr, const std::vector<Token>& tokens) {
    // Known macro keyword
    if (macroAttrs.find(attr.name) == macroAttrs.end()) {
        return false;
    }

    // Single string parameter
    if (attr.tok_indecies.size() != 5 || !tokens[attr.tok_indecies[3]].is(tok::string_literal)) {
        return false;
    }

    return true;
}

void removeOklAttr(const std::vector<Token>& tokens,
                   const OklAttribute& attr,
                   oklt::Rewriter& rewriter) {
    // remove OKL specific attribute in source code
    SourceLocation attrLocStart(tokens[attr.tok_indecies.front()].getLocation());
    SourceLocation attrLocEnd(tokens[attr.tok_indecies.back()].getLastLoc());
    SourceRange attrSrcRange(attrLocStart, attrLocEnd);
    auto removedStr = rewriter.getRewrittenText(attrSrcRange);
    rewriter.RemoveText(attrSrcRange);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// routine to replace OKL macro attribute with direct macro
bool replaceOklMacroAttribute(const OklAttribute& oklAttr,
                              const std::vector<Token>& tokens,
                              Preprocessor& pp,
                              oklt::Rewriter& rewriter) {
    if (!isProbablyMacroAttr(oklAttr, tokens)) {
        return true;
    }

    // TODO log each modification to adjust marker line col coordinate accordingly
    removeOklAttr(tokens, oklAttr, rewriter);

    auto insertLoc(tokens[oklAttr.tok_indecies.front()].getLocation());

    SmallVector<Token, 1> argToks = {tokens[oklAttr.tok_indecies[3]]};
    auto lit = StringLiteralParser(argToks, pp);
    if (lit.hadError) {
        return false;
    }

    auto replacement = lit.GetString().str();

    // in case of one line source code add new line character before and after directive expand to
    // ensure source is not hidden by macro that could be inside of directive
    const auto& sm = pp.getSourceManager();
    if (oklAttr.tok_indecies.back() != tokens.size()) {
        auto& rightEdge = oklAttr.tok_indecies.back();

        FullSourceLoc attrLastTokLoc(tokens[rightEdge].getLocation(), sm);
        FullSourceLoc nextTokLoc(tokens[rightEdge + 1].getLocation(), sm);

        if (attrLastTokLoc.getExpansionLineNumber() == nextTokLoc.getExpansionLineNumber()) {
            replacement += '\n';
        }
    }
    if (oklAttr.tok_indecies.front() != 0) {
        auto& leftEdge = oklAttr.tok_indecies.back();

        FullSourceLoc attrFirtTokLoc(tokens[leftEdge].getLocation(), sm);
        FullSourceLoc prevTokLoc(tokens[leftEdge - 1].getLocation(), sm);

        if (attrFirtTokLoc.getExpansionLineNumber() == prevTokLoc.getExpansionLineNumber()) {
            replacement = '\n' + replacement;
        }
    }

    rewriter.InsertTextBefore(insertLoc, replacement);

    SPDLOG_DEBUG("Removed macro attr: {} at loc: {}", oklAttr.name, insertLoc.printToString(sm));

    return true;
}

class OklDirectiveExpansion : public StageAction {
   public:
    OklDirectiveExpansion() { _name = OKL_DIRECTIVE_EXPANSION_STAGE; };

    bool BeginSourceFileAction(clang::CompilerInstance& compiler) override {
        auto& pp = compiler.getPreprocessor();
        auto tokens = fetchTokens(pp);

        if (tokens.empty()) {
            _stage->pushError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING, "no tokens in source?");
            return false;
        }

        auto& rewriter = _stage->getRewriter();
        auto result = visitOklAttributes(
            tokens,
            pp,
            [this, &rewriter](
                const OklAttribute& attr, const std::vector<Token>& tokens, Preprocessor& pp) {
                replaceOklMacroAttribute(attr, tokens, pp, rewriter);
                return true;
            });
        if (!result) {
            _stage->pushError(result.error().ec, result.error().desc);
            return false;
        }

        return true;
    }
};

StagePluginRegistry::Add<OklDirectiveExpansion> oklDirectiveExpansion(OKL_DIRECTIVE_EXPANSION_STAGE,
                                                                      "");
}  // namespace
