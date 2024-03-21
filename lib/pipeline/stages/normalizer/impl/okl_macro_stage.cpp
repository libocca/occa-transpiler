#include <oklt/core/error.h>

#include "core/transpiler_session/session_stage.h"
#include "core/vfs/overlay_fs.h"

#include "pipeline/stages/normalizer/error_codes.h"
#include "pipeline/stages/normalizer/impl/okl_attr_traverser.h"
#include "pipeline/stages/normalizer/impl/okl_macro_stage.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Lex/Lexer.h>
#include <clang/Lex/LiteralSupport.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Tooling.h>

#include <set>

namespace {

using namespace clang;
using namespace oklt;

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

void removeOklAttr(const std::vector<Token>& tokens, const OklAttribute& attr, Rewriter& rewriter) {
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
                              Rewriter& rewriter) {
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

    rewriter.InsertTextBefore(insertLoc, lit.GetString());

#ifdef NORMALIZER_DEBUG_LOG
    llvm::outs() << "removed macro attr: " << oklAttr.name
                 << " at loc: " << insertLoc.printToString(pp.getSourceManager()) << '\n';
#endif

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
            if (spelling.empty() || spelling[0] != OKL_ATTR_MARKER) {
                break;
            }
            tok.setKind(tok::at);
        }

        tokens.push_back(tok);
    }

    return tokens;
}

struct OklMacroAttributeNormalizerAction : public clang::ASTFrontendAction {
    explicit OklMacroAttributeNormalizerAction(OklMacroStageInput& input,
                                               OklMacroStageOutput& output)
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
        pp.EnterMainSourceFile();
        auto tokens = fetchTokens(pp);

        if (tokens.empty()) {
            _session.pushError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING,
                               "no tokens in source?");
            return false;
        }

        SessionStage stage{_session, compiler};
        auto& rewriter = stage.getRewriter();

        auto result = visitOklAttributes(
            tokens,
            pp,
            [this, &rewriter](
                const OklAttribute& attr, const std::vector<Token>& tokens, Preprocessor& pp) {
                replaceOklMacroAttribute(attr, tokens, pp, rewriter);
                return true;
            });
        if (!result) {
            _session.pushError(result.error().ec, result.error().desc);
            return false;
        }

        _output.cppSrc = stage.getRewriterResultForMainFile();
        _output.cppIncs = stage.getRewriterResultForHeaders();

        pp.EndSourceFile();

        return false;
    }

   private:
    OklMacroStageInput& _input;
    OklMacroStageOutput& _output;
    TranspilerSession& _session;
};
}  // namespace
namespace oklt {

OklMacroResult convertOklMacroAttribute(OklMacroStageInput input) {
    if (input.cppSrc.empty()) {
        llvm::outs() << "input source string is empty\n";
        auto error =
            makeError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING, "input source string is empty");
        return tl::make_unexpected(std::vector<Error>{error});
    }

#ifdef NORMALIZER_DEBUG_LOG
    llvm::outs() << "stage 0 OKL source:\n\n" << input.oklCppSrc << '\n';
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

    OklMacroStageOutput output = {.session = input.session};
    auto ok = tooling::runToolOnCodeWithArgs(
        std::make_unique<OklMacroAttributeNormalizerAction>(input, output),
        input_file,
        args,
        file_name,
        tool_name);
    if (!ok) {
        return tl::make_unexpected(std::move(output.session->getErrors()));
    }

    // no errors and empty output could mean that the source is already normalized
    // so use input as output and lets the next stage try to figure out
    if (output.cppSrc.empty()) {
        output.cppSrc = std::move(input_file);
    }

#ifdef NORMALIZER_DEBUG_LOG
    llvm::outs() << "stage 0 Macro cpp source:\n\n" << output.cppSrc << '\n';
#endif

    return output;
}
}  // namespace oklt
