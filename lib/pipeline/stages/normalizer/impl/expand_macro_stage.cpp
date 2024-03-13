#include <oklt/core/error.h>

#include "core/transpiler_session/session_stage.h"

#include "pipeline/stages/normalizer/error_codes.h"
#include "pipeline/stages/normalizer/impl/expand_macro_stage.h"
#include "pipeline/stages/normalizer/impl/okl_attr_traverser.h"
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

void expandAndInlineMacroWithOkl(Preprocessor& pp, SessionStage& stage) {
    auto ctx = std::make_unique<MacroExpansionContext>(pp.getLangOpts());
    ctx->registerForPreprocessor(pp);

    pp.EnterMainSourceFile();
    const auto& sm = pp.getSourceManager();
    auto& rewriter = stage.getRewriter();

    while (true) {
        Token tok{};
        pp.Lex(tok);

        if (tok.is(tok::eof)) {
            break;
        }

        // catch start of macro expension
        if (!Lexer::isAtStartOfMacroExpansion(
                tok.getLocation(), pp.getSourceManager(), pp.getLangOpts())) {
            continue;
        }

        // and it's is in user file
        auto fid = sm.getFileID(tok.getLocation());
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

        // TODO add options?
        // if (!expanded->contains(OKL_ATTR_MARKER)) {
        //     continue;
        // }

        auto orginal = ctx->getOriginalText(expansionLoc);
        if (!orginal) {
            llvm::outs() << "no original macro under: " << expansionLoc.printToString(sm) << '\n';
            continue;
        }

#ifdef NORMALIZER_DEBUG_LOG
        llvm::outs() << "at " << expansionLoc.printToString(sm) << " inline macro "
                     << orginal.value() << " by " << expanded.value() << '\n';
#endif
        rewriter.ReplaceText(expansionLoc, orginal->size(), expanded.value());
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
