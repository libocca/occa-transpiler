#include "gnu_to_std_cpp_stage.h"

#include <oklt/core/diag/diag_consumer.h>
#include <oklt/core/error.h>
#include <oklt/core/transpiler_session/session_stage.h>
#include <oklt/pipeline/stages/normalizer/error_codes.h>

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Tooling.h>

using namespace oklt;
using namespace clang;

namespace {
struct AttrNormalizerCtx {
    ASTContext* astCtx;
    Rewriter* rewriter;
    std::list<OklAttrMarker> markers;
};

// TODO move to helper functions header
constexpr int32_t CXX11_ATTR_PREFIX_LEN = 2;
constexpr int32_t CXX11_ATTR_SUFFIX_LEN = 2;

constexpr int32_t GNU_ATTR_PREFIX_LEN = 15;
constexpr int32_t GNU_ATTR_SUFFIX_LEN = 2;

SourceRange getAttrFullSourceRange(const Attr& attr) {
    auto arange = attr.getRange();

    if (attr.isCXX11Attribute() || attr.isC2xAttribute()) {
        arange.setBegin(arange.getBegin().getLocWithOffset(-CXX11_ATTR_PREFIX_LEN));
        arange.setEnd(arange.getEnd().getLocWithOffset(CXX11_ATTR_SUFFIX_LEN));
    }

    if (attr.isGNUAttribute()) {
        arange.setBegin(arange.getBegin().getLocWithOffset(-GNU_ATTR_PREFIX_LEN));
        arange.setEnd(arange.getEnd().getLocWithOffset(GNU_ATTR_SUFFIX_LEN));
    }

    return arange;
}

void removeAttr(Rewriter& rewriter, const Attr& attr) {
    auto arange = getAttrFullSourceRange(attr);
    rewriter.RemoveText(arange);
}

const char* OKL_PREFIX = "okl_";
bool hasAttrOklPrefix(const Attr& attr) {
    return attr.getAttrName()->getName().startswith(OKL_PREFIX);
}

std::string getArgAsStr(const SuppressAttr& attr) {
    auto str = attr.diagnosticIdentifiers_begin()->data();
    return str != nullptr ? str : "";
}

std::string getOklName(const Attr& attr) {
    return attr.getAttrName()->getName().split('_').second.str();
}

OklAttribute toOklAttr(const AnnotateAttr& attr, ASTContext& ast) {
    return OklAttribute{.raw = "",
                        .name = getOklName(attr),
                        .params = attr.getAnnotation().str(),
                        .tok_indecies = {}};
}

OklAttribute toOklAttr(const SuppressAttr& attr, ASTContext& ast) {
    assert(attr.diagnosticIdentifiers_size() != 0 && "suppress attr has 0 args");
    return OklAttribute{
        .raw = "", .name = getOklName(attr), .params = getArgAsStr(attr), .tok_indecies = {}};
}

template <typename Expr, typename AttrType>
void insertNormalizedAttr(const Expr& e, const AttrType& attr, SessionStage& stage) {
    auto oklAttr = toOklAttr(attr, stage.getCompiler().getASTContext());
    auto normalizedAttrStr = wrapAsSpecificCxxAttr(oklAttr);
    stage.getRewriter().InsertTextAfter(e.getBeginLoc(), normalizedAttrStr);
}

template <typename AttrType, typename Expr>
bool tryToNormalizeAttrExpr(Expr& e, SessionStage& stage, const Attr** lastProccesedAttr) {
    assert(lastProccesedAttr);
    for (auto* attr : e.getAttrs()) {
        if ((*lastProccesedAttr) && ((*lastProccesedAttr)->getLoc() == attr->getLoc())) {
            continue;
        }

        if (attr->isC2xAttribute() || attr->isCXX11Attribute()) {
            continue;
        }

        if (!hasAttrOklPrefix(*attr)) {
            continue;
        }

        const auto* targetAttr = dyn_cast_or_null<AttrType>(attr);
        if (!targetAttr) {
            continue;
        }

        removeAttr(stage.getRewriter(), *attr);
        insertNormalizedAttr(e, *targetAttr, stage);
        *lastProccesedAttr = attr;
    }

    return true;
}

SourceLocation getMarkerSourceLoc(const OklAttrMarker& marker, const SourceManager& srcMng) {
    return srcMng.translateLineCol(srcMng.getMainFileID(), marker.loc.line, marker.loc.col);
}

// Traverse AST and normalize GMU attributes and fix markers to standard C++ attribute
// representation
class GnuToCppAttrNormalizer : public RecursiveASTVisitor<GnuToCppAttrNormalizer> {
   public:
    explicit GnuToCppAttrNormalizer(SessionStage& stage)
        : _stage(stage) {
        auto anyCtx = _stage.getUserCtx("input");
        if (anyCtx && anyCtx->has_value()) {
            // use non-throw api by passing pointer to any
            _input = *(std::any_cast<GnuToStdCppStageInput*>(anyCtx));
        } else {
            _input = nullptr;
        }
    }

    bool VisitDecl(Decl* d) {
        assert(d != nullptr && "declaration is nullptr");

        if (!d->hasAttrs()) {
            return true;
        }
        return tryToNormalizeAttrExpr<AnnotateAttr>(*d, _stage, &_lastProccesedAttr);
    }

    bool TraverseAttributedStmt(AttributedStmt* as) {
        assert(as != nullptr && "attributed statement is nullptr");

        if (!tryToNormalizeAttrExpr<SuppressAttr>(*as, _stage, &_lastProccesedAttr)) {
            return false;
        }

        return RecursiveASTVisitor<GnuToCppAttrNormalizer>::TraverseAttributedStmt(as);
    }

    // Special visitor for attribute inside in 'for loop' statement
    bool VisitForStmt(ForStmt* s) {
        assert(s != nullptr && "ForStmt is null");
        assert(_input != nullptr && "input is null");

        if (_input->recoveryMarkers.empty()) {
            return true;
        }

        const auto& marker = _input->recoveryMarkers.front();
        auto markerLoc = getMarkerSourceLoc(marker, _stage.getCompiler().getSourceManager());
        auto forParenRange = SourceRange(s->getBeginLoc(), s->getRParenLoc());

#ifdef NORMALIZER_DEBUG_LOG
        llvm::outs() << "for loc: "
                     << forParenRange.printToString(_stage.getCompiler().getSourceManager())
                     << "\nmarker loc: "
                     << markerLoc.printToString(_stage.getCompiler().getSourceManager()) << '\n';
#endif

        // if marker is inside of loop source location range it indicates OKL loop that should be decorated
        // by attribute in marker
        if (forParenRange.getBegin() <= markerLoc && markerLoc <= forParenRange.getEnd()) {
            _stage.getRewriter().InsertTextBefore(forParenRange.getBegin(),
                                                  wrapAsSpecificCxxAttr(marker.attr));
            _input->recoveryMarkers.pop_front();
        }

        return true;
    }

   private:
    const Attr* _lastProccesedAttr{nullptr};
    SessionStage& _stage;
    GnuToStdCppStageInput* _input;
};

// ASTConsumer to run GNU to C++ attribute replacing
class GnuToCppAttrNormalizerConsumer : public ASTConsumer {
   public:
    explicit GnuToCppAttrNormalizerConsumer(SessionStage& stage)
        : _stage(stage),
          _normalizer_visitor(_stage) {}

    // Override the method that gets called for each parsed top-level
    // declaration.
    void HandleTranslationUnit(ASTContext& ctx) override {
#ifdef NORMALIZER_DEBUG_LOG
        ctx.getTranslationUnitDecl()->dump(llvm::outs());
#endif
        TranslationUnitDecl* decl = ctx.getTranslationUnitDecl();
        _normalizer_visitor.TraverseDecl(decl);
    }

   private:
    SessionStage& _stage;
    GnuToCppAttrNormalizer _normalizer_visitor;
};

struct GnuToStdCppAttributeNormalizerAction : public clang::ASTFrontendAction {
    explicit GnuToStdCppAttributeNormalizerAction(oklt::GnuToStdCppStageInput& input,
                                                  oklt::GnuToStdCppStageOutput& output)
        : _input(input),
          _output(output),
          _session(*input.session),
          _stage(nullptr) {}

   protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                          llvm::StringRef in_file) override {
        _stage = std::make_unique<SessionStage>(_session, compiler);
        if (!_stage->setUserCtx("input", &_input)) {
            _stage->pushError(std::error_code(),
                              "failed to set user ctx for GnuToStdCppAttributeNormalizerAction");
            return nullptr;
        }
        auto consumer = std::make_unique<GnuToCppAttrNormalizerConsumer>(*_stage);
        compiler.getDiagnostics().setClient(new DiagConsumer(*_stage));
        return std::move(consumer);
    }

    void EndSourceFileAction() override {
        if (!_stage) {
            _stage->pushError(std::error_code(), "where is my stage???");
            return;
        }

        _output.stdCppSrc = _stage->getRewriterResult();
    }

   private:
    oklt::GnuToStdCppStageInput& _input;
    oklt::GnuToStdCppStageOutput& _output;
    TranspilerSession& _session;
    std::unique_ptr<SessionStage> _stage;
};

}  // namespace

namespace oklt {
GnuToStdCppResult convertGnuToStdCppAttribute(GnuToStdCppStageInput input) {
    if (input.gnuCppSrc.empty()) {
        llvm::outs() << "input source string is empty\n";
        auto error =
            makeError(OkltNormalizerErrorCode::EMPTY_SOURCE_STRING, "input source string is empty");
        return tl::make_unexpected(std::vector<Error>{error});
    }

    Twine tool_name = "okl-transpiler-normalization-to-cxx";
    Twine file_name("gnu-kernel-to-cxx.cpp");
    std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

    auto input_file = std::move(input.gnuCppSrc);
    GnuToStdCppStageOutput output = {.session = input.session};
    auto ok = tooling::runToolOnCodeWithArgs(
        std::make_unique<GnuToStdCppAttributeNormalizerAction>(input, output),
        input_file,
        args,
        file_name,
        tool_name);

    if (!ok) {
        return tl::make_unexpected(std::move(output.session->getErrors()));
    }

    // no errors and empty output could mean that the source is already normalized
    // so use input as output and lets the next stage try to figure out
    if (output.stdCppSrc.empty()) {
        output.stdCppSrc = std::move(input_file);
    }

#ifdef NORMALIZER_DEBUG_LOG
    llvm::outs() << "stage 2 STD cpp source:\n\n" << output.stdCppSrc << '\n';
#endif

    return output;
}
}  // namespace oklt
