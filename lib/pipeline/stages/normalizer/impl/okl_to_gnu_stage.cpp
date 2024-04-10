#include <oklt/core/error.h>

#include "attributes/attribute_names.h"
#include "core/rewriter/impl/dtree_rewriter_proxy.h"
#include "core/transpiler_session/session_stage.h"
#include "core/vfs/overlay_fs.h"

#include "pipeline/stages/normalizer/error_codes.h"
#include "pipeline/stages/normalizer/impl/okl_attr_traverser.h"
#include "pipeline/stages/normalizer/impl/okl_to_gnu_stage.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Analysis/MacroExpansionContext.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/Tooling.h>
#include <spdlog/spdlog.h>

namespace {

using namespace clang;
using namespace oklt;

bool isProbablyOklSpecificForStmt(Token left, Token right) {
    return left.is(tok::semi) && right.is(tok::r_paren);
}

bool isProbablyAtBeginnigOfExpr(Token left, Token right) {
    return ((left.is(tok::semi) || left.is(tok::l_brace)) && !right.is(tok::semi));
}

Token getLeftNeigbour(const OklAttribute& attr, const std::vector<Token>& tokens) {
    return attr.tok_indecies.front() != 0 ? tokens[attr.tok_indecies.front() - 1] : Token();
}

Token getRightNeigbour(const OklAttribute& attr, const std::vector<Token>& tokens) {
    return attr.tok_indecies.back() != tokens.size() ? tokens[attr.tok_indecies.back() + 1]
                                                     : Token();
}

OklAttrMarker makeOklAttrMarker(const Preprocessor& pp,
                                const OklAttribute& oklAtr,
                                const SourceLocation loc) {
    auto isValid = loc.isValid();
    return {.attr = oklAtr,
            .loc = {.line = pp.getSourceManager().getPresumedLineNumber(loc),
                    .col = pp.getSourceManager().getPresumedColumnNumber(loc)}};
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
bool replaceOklByGnuAttribute(std::list<OklAttrMarker>& gnu_markers,
                              const OklAttribute& oklAttr,
                              const std::vector<Token>& tokens,
                              Preprocessor& pp,
                              oklt::Rewriter& rewriter,
                              TranspilerSession& session) {
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
    auto& mapper = session.getOriginalSourceMapper();
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
    // handle all cases) than standard attribute. After parsing AST GNU will be replaced by CXX NB:
    // there are cases where GNU attribute could not be parsed and embed into AST For example:
    //   a+=1 __attrbute((okl_atomic));
    // and
    //   expected unqualified-id
    // error will be generated by clang AST parser
    else {
        auto gnuAttr = wrapAsSpecificGnuAttr(oklAttr);
        rewriter.ReplaceText(attrSrcRange, gnuAttr);
        gnu_markers.emplace_back(makeOklAttrMarker(pp, oklAttr, insertLoc));
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

std::vector<Token> fetchTokens(Preprocessor& pp) {
    std::vector<Token> tokens;
    while (true) {
        Token tok{};
        pp.Lex(tok);

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
    }

    return tokens;
}

struct OklToGnuAttributeNormalizerAction : public clang::ASTFrontendAction {
    explicit OklToGnuAttributeNormalizerAction(OklToGnuStageInput& input,
                                               OklToGnuStageOutput& output)
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
        if (compiler.hasFileManager()) {
            auto overlayFs = makeOverlayFs(compiler.getFileManager().getVirtualFileSystemPtr(),
                                           _input.oklCppIncs);
            compiler.getFileManager().setVirtualFileSystem(overlayFs);
        }

        auto& pp = compiler.getPreprocessor();
        pp.EnterMainSourceFile();
        auto tokens = fetchTokens(pp);

        if (tokens.empty()) {
            _session.pushError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING,
                               "no tokens in source?");
            return false;
        }

        SessionStage stage{_session, compiler, RewriterProxyType::WithDeltaTree};
        auto& rewriter = stage.getRewriter();

        auto result = visitOklAttributes(
            tokens,
            pp,
            [this, &rewriter](
                const OklAttribute& attr, const std::vector<Token>& tokens, Preprocessor& pp) {
                replaceOklByGnuAttribute(_output.gnuMarkers, attr, tokens, pp, rewriter, _session);
                return true;
            });
        if (!result) {
            _session.pushError(result.error().ec, result.error().desc);
            return false;
        }

        _output.gnuCppSrc = stage.getRewriterResultForMainFile();
        // no errors and empty output could mean that the source is already normalized
        // so use input as output and lets the next stage try to figure out
        if (_output.gnuCppSrc.empty()) {
            _output.gnuCppSrc = std::move(_input.oklCppSrc);
        }

        // we need keep all headers in output even there are not modififcation by rewriter to
        // populate affected files futher
        _output.gnuCppIncs = stage.getRewriterResultForHeaders();
        _output.gnuCppIncs.fileMap.merge(_input.oklCppIncs.fileMap);

        pp.EndSourceFile();

        return false;
    }

   private:
    OklToGnuStageInput& _input;
    OklToGnuStageOutput& _output;
    TranspilerSession& _session;
};
}  // namespace
namespace oklt {

OklToGnuResult convertOklToGnuAttribute(OklToGnuStageInput input) {
    if (input.oklCppSrc.empty()) {
        SPDLOG_ERROR("Input source string is empty");
        auto error =
            makeError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING, "input source string is empty");
        return tl::make_unexpected(std::vector<Error>{error});
    }

    SPDLOG_DEBUG("stage 0 OKL source:\n\n{}", input.oklCppSrc);

    Twine tool_name = "okl-transpiler-normalization-to-gnu";
    auto cppFileNamePath = input.session->input.sourcePath;
    auto cppFileName = std::string(cppFileNamePath.replace_extension(".cpp"));
    std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

    auto input_file = std::move(input.oklCppSrc);

    auto& sessionInput = input.session->input;
    for (const auto& define : sessionInput.defines) {
        std::string def = "-D" + define;
        args.push_back(std::move(def));
    }

    for (const auto& includePath : sessionInput.inlcudeDirectories) {
        std::string incPath = "-I" + includePath.string();
        args.push_back(std::move(incPath));
    }

    OklToGnuStageOutput output = {.session = input.session};
    auto ok = tooling::runToolOnCodeWithArgs(
        std::make_unique<OklToGnuAttributeNormalizerAction>(input, output),
        input_file,
        args,
        cppFileName,
        tool_name);
    if (!ok) {
        return tl::make_unexpected(std::move(output.session->getErrors()));
    }

    // no errors and empty output could mean that the source is already normalized
    // so use input as output and lets the next stage try to figure out
    if (output.gnuCppSrc.empty()) {
        output.gnuCppSrc = std::move(input_file);
    }

    SPDLOG_DEBUG("stage 1 GNU cpp source:\n\n{}", output.gnuCppSrc);

    return output;
}
}  // namespace oklt
