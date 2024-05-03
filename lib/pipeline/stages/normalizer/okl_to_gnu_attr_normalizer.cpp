#include "core/lex/lexer.h"
#include "core/transpiler_session/session_stage.h"

#include "pipeline/core/error_codes.h"
#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_registry.h"

#include "pipeline/utils/okl_attribute.h"
#include "pipeline/utils/okl_attribute_traverser.h"

#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/LiteralSupport.h>

#include <spdlog/spdlog.h>

namespace {
using namespace clang;
using namespace oklt;

bool isProbablyOklSpecificForStmt(Token left, Token right) {
    return left.is(tok::semi) && right.is(tok::r_paren);
}

bool isProbablyAtBeginnigOfExpr(Token left, Token right) {
    return (left.isOneOf(tok::semi, tok::l_brace, tok::r_paren) &&
            !right.isOneOf(tok::semi, tok::r_paren));
}

Token getLeftNeigbour(const OklAttribute& attr, const std::vector<Token>& tokens) {
    return attr.tok_indecies.front() != 0 ? tokens[attr.tok_indecies.front() - 1] : Token();
}

Token getRightNeigbour(const OklAttribute& attr, const std::vector<Token>& tokens) {
    return attr.tok_indecies.back() != tokens.size() ? tokens[attr.tok_indecies.back() + 1]
                                                     : Token();
}

SourceLocation findForKwLocBefore(const std::vector<Token> tokens, size_t start) {
    for (size_t i = start; i != 0; --i) {
        const auto& tok = tokens.at(i);
        if (tok.is(tok::kw_for)) {
            return tok.getLocation();
        }
    }
    return SourceLocation();
}

uint32_t getTokenOffset(const Token& tok, const Preprocessor& pp) {
    return pp.getSourceManager().getFileOffset(tok.getLocation());
}

std::pair<FileID, uint32_t> getTokenFidLineNumber(const Token& tok, const Preprocessor& pp) {
    return {pp.getSourceManager().getFileID(tok.getLocation()),
            pp.getSourceManager().getSpellingLineNumber(tok.getLocation())};
}

uint32_t gettokenColNumber(const Token& tok, const Preprocessor& pp) {
    return pp.getSourceManager().getSpellingColumnNumber(tok.getLocation());
}

std::string getTokenLine(const Token& tok, const Preprocessor& pp) {
    auto& SM = pp.getSourceManager();
    auto [fileID, lineNumber] = getTokenFidLineNumber(tok, pp);
    llvm::StringRef bufferData = SM.getBufferData(fileID);
    auto locStart = SM.translateLineCol(fileID, lineNumber, 1);
    auto startOffset = SM.getFileOffset(locStart);
    auto endOffset = bufferData.find('\n', startOffset);  // FIXME: is that portable?
    if (endOffset == llvm::StringRef::npos) {
        endOffset = bufferData.size();
    }
    return std::string(bufferData.substr(startOffset, endOffset - startOffset));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// routine to replace OKL attribute with GNU one and store it original source location
// one trick is that functions could fix malformed C++ for statement with extra semi
bool replaceOklByGnuAttribute(const OklAttribute& oklAttr,
                              const std::vector<Token>& tokens,
                              Preprocessor& pp,
                              oklt::Rewriter& rewriter,
                              OriginalSourceMapper& mapper) {
    // TODO log each modification to adjust marker line col coordinate accordingly
    auto& attrBegToken = tokens[oklAttr.tok_indecies.front()];
    if (attrBegToken.getLocation().isInvalid()) {
        SPDLOG_ERROR("Invalid attribute token location");
        return false;
    }
    auto oklAttrOffset = getTokenOffset(attrBegToken, pp);
    auto oklAttrColNumber = gettokenColNumber(attrBegToken, pp);
    auto attrFidLine = getTokenFidLineNumber(attrBegToken, pp);

    // Insert original line to the originalLines mapping if needed
    auto attrLine = getTokenLine(attrBegToken, pp);
    mapper.addOriginalLine(attrFidLine, attrLine);

    auto leftNeighbour = getLeftNeigbour(oklAttr, tokens);
    auto rightNeighbour = getRightNeigbour(oklAttr, tokens);
    auto insertLoc(tokens[oklAttr.tok_indecies.front()].getLocation());

    SourceLocation attrLocStart(tokens[oklAttr.tok_indecies.front()].getLocation());
    SourceLocation attrLocEnd(tokens[oklAttr.tok_indecies.back()].getLastLoc());
    SourceRange attrSrcRange(attrLocStart, attrLocEnd);
    int attrOffsetFromBeginToName = GNU_ATTRIBUTE_BEGIN_TO_NAME_OFFSET;  // 2 for CXX, 15 for GNU

    // fix malformed C++ syntax like for(init;cond;step;@outer) to [[okl::outer]]
    // for(init;cond;step) we assume that attribute is inside of for loop and 'for' keyword is
    // definitely before attribute
    if (isProbablyOklSpecificForStmt(leftNeighbour, rightNeighbour)) {
        rewriter.ReplaceText(leftNeighbour.getLocation(), 1, ")");
        rewriter.ReplaceText(rightNeighbour.getLocation(), 1, " ");

        // TODO delegate parsing of 'for' statement to okl attribute parser
        auto forLoc = findForKwLocBefore(tokens, oklAttr.tok_indecies.front());
        if (forLoc.isInvalid()) {
            SPDLOG_ERROR("no kw_for is found before loc: {}\n",
                         leftNeighbour.getLocation().printToString(pp.getSourceManager()));
            return false;
        }
        auto gnuAttr = wrapAsSpecificGnuAttr(oklAttr);
        rewriter.InsertTextBefore(forLoc, gnuAttr);
        rewriter.RemoveText(attrSrcRange);
        insertLoc = forLoc;
    }
    // INFO: just replace directly with standard attribute
    // if it's originally at the beginning, or an in-place type attribute.
    else if (isProbablyAtBeginnigOfExpr(leftNeighbour, rightNeighbour)) {
        auto cppAttr = wrapAsSpecificCxxAttr(oklAttr);
        attrOffsetFromBeginToName = CXX_ATTRIBUTE_BEGIN_TO_NAME_OFFSET;
        rewriter.ReplaceText(attrSrcRange, cppAttr);
    }
    // INFO: attribute is not at the beginning of expr so wrap it as GNU.
    // GNU attribute has more diversity for locations (and it's a nightmare for parser and AST to
    // handle all cases) than standard attribute. After parsing AST GNU will be replaced by CXX
    // NB:
    // there are cases where GNU attribute could not be parsed and embed into AST For example:
    //   a+=1 __attrbute((okl_atomic));
    // and
    //   expected unqualified-id
    // error will be generated by clang AST parser
    else {
        auto gnuAttr = wrapAsSpecificGnuAttr(oklAttr);
        rewriter.ReplaceText(attrSrcRange, gnuAttr);
    }

    // Save offset to original column mapping
    if (!mapper.addAttributeColumn(
            insertLoc, oklAttrColNumber, rewriter, attrOffsetFromBeginToName)) {
        SPDLOG_ERROR("OKL to GNU attribute stage expected Rewriter with DeltaTrees");
    }

    SPDLOG_DEBUG("removed attr: {} at loc: {}",
                 oklAttr.name,
                 insertLoc.printToString(pp.getSourceManager()));

    return true;
}

class OklToGnuAttrNormalizer : public StageAction {
   public:
    OklToGnuAttrNormalizer() { _name = OKL_TO_GNU_ATTR_NORMALIZER_STAGE; }

    bool BeginSourceFileAction(clang::CompilerInstance& compiler) override {
        auto& pp = compiler.getPreprocessor();
        auto tokens = fetchTokens(pp);

        if (tokens.empty()) {
            _stage->pushError(OkltPipelineErrorCode::EMPTY_SOURCE_STRING, "no tokens in source?");
            return false;
        }

        auto& rewriter = _stage->getRewriter();
        auto result = visitOklAttributes(
            tokens,
            pp,
            [this, &rewriter](
                const OklAttribute& attr, const std::vector<Token>& tokens, Preprocessor& pp) {
                return replaceOklByGnuAttribute(
                    attr, tokens, pp, rewriter, _stage->getSession().getOriginalSourceMapper());
            });
        if (!result) {
            _stage->pushError(result.error().ec, result.error().desc);
            return false;
        }

        return true;
    }

   protected:
    RewriterProxyType getRewriterType() const override { return RewriterProxyType::WithDeltaTree; }
};

StagePluginRegistry::Add<OklToGnuAttrNormalizer> oklToGnuAttrNormalizer(
    OKL_TO_GNU_ATTR_NORMALIZER_STAGE,
    "");
}  // namespace
