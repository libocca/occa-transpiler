#include "gnu_to_std_cpp_stage.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Rewrite/Core/Rewriter.h>
#include <clang/Tooling/Tooling.h>

using namespace oklt;
using namespace clang;

// #define DEBUG_NORMALIZER
//
namespace {
struct AttrNormalizerCtx {
  ASTContext* astCtx;
  Rewriter* rewriter;
  std::list<OklAttrMarker> markers;
};

std::string wrapAsSpecificCxxAttr(const OklAttribute& attr) {
  if (attr.params.empty()) {
    return "[[okl::" + attr.name + R"((")" + "(void)" + "\")]]";
  }

  return "[[okl::" + attr.name + R"((")" + attr.params + "\")]]";
}
// TODO move to helper functions header
constexpr uint32_t CXX11_ATTR_PREFIX_LEN = 2u;
constexpr uint32_t CXX11_ATTR_SUFFIX_LEN = 2u;

constexpr uint32_t GNU_ATTR_PREFIX_LEN = 15u;
constexpr uint32_t GNU_ATTR_SUFFIX_LEN = 2u;
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

OklAttribute toOklAttr(const AnnotateAttr& attr, ASTContext& ast) {
  return OklAttribute{.raw          = "",
                      .name         = attr.getAttrName()->getName().split('_').second.str(),
                      .params       = attr.getAnnotation().str(),
                      .begin_loc    = SourceLocation(),
                      .tok_indecies = {}};
}

OklAttribute toOklAttr(const SuppressAttr& attr, ASTContext& ast) {
  assert(attr.diagnosticIdentifiers_size() != 0 && "suppress attr has 0 args");

  const auto* args_str = attr.diagnosticIdentifiers_begin()->data();
  return OklAttribute{.raw          = "",
                      .name         = attr.getAttrName()->getName().split('_').second.str(),
                      .params       = args_str,
                      .begin_loc    = SourceLocation(),
                      .tok_indecies = {}};
}

template <typename Expr, typename AttrType>
void insertNormalizedAttr(const Expr& e, const AttrType& attr, SessionStage& stage) {
  auto oklAttr           = toOklAttr(attr, stage.getCompiler().getASTContext());
  auto normalizedAttrStr = wrapAsSpecificCxxAttr(oklAttr);
  stage.getRewriter().InsertTextBefore(e.getBeginLoc(), normalizedAttrStr);
}

template <typename AttrType, typename Expr>
bool tryToNormalizeAttrExpr(Expr& e, SessionStage& stage) {
  for (const auto attr : e.getAttrs()) {
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
  }

  return true;
}

// Traverse AST and normalize GMU attributes and fix markers to standard C++ attribute
// representation
class GnuToCppAttrNormalizer : public RecursiveASTVisitor<GnuToCppAttrNormalizer> {
 public:
  explicit GnuToCppAttrNormalizer(SessionStage& stage) : _stage(stage) {
    auto anyCtx = _stage.getUserCtx("input");
    if (anyCtx.has_value()) {
      // use non-throw api by passing pointer to any
      _input = *(std::any_cast<GnuToStdCppStageInput*>(&anyCtx));
    } else {
      _input = nullptr;
    }
  }

  bool VisitDecl(Decl* d) {
    assert(d != nullptr && "declaration is nullptr");

    return tryToNormalizeAttrExpr<AnnotateAttr>(*d, _stage);
  }

  bool TraverseAttributedStmt(AttributedStmt* as) {
    assert(as != nullptr && "attributed statement is nullptr");

    if (!tryToNormalizeAttrExpr<SuppressAttr>(*as, _stage)){
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
    auto s_range       = s->getSourceRange();
    // if marker is inside of loop source location range it means loop should be decorated
    // by attribute in marker
    if (s_range.getBegin() <= marker.loc || marker.loc <= s_range.getEnd()) {
      _stage.getRewriter().InsertTextBefore(s_range.getBegin(),
                                            wrapAsSpecificCxxAttr(marker.attr));
      _input->recoveryMarkers.pop_front();
    }

    return true;
  }

 private:
  SessionStage& _stage;
  GnuToStdCppStageInput* _input;
};

// ASTConsumer to run GNU to C++ attribute replacing
class GnuToCppAttrNormalizerConsumer : public ASTConsumer {
 public:
  explicit GnuToCppAttrNormalizerConsumer(SessionStage& stage)
      : _normalizer_visitor(stage), _stage(stage) {}

  // Override the method that gets called for each parsed top-level
  // declaration.
  void HandleTranslationUnit(ASTContext& ctx) override {
#ifdef DEBUG_NORMALIZER
    ctx.getTranslationUnitDecl()->dump(llvm::outs());
#endif
    TranslationUnitDecl* decl = ctx.getTranslationUnitDecl();
    _normalizer_visitor.TraverseDecl(decl);
  }

 private:
  GnuToCppAttrNormalizer _normalizer_visitor;
  SessionStage& _stage;
};

struct GnuToStdCppAttributeNormalizerAction : public clang::ASTFrontendAction {
  explicit GnuToStdCppAttributeNormalizerAction(oklt::GnuToStdCppStageInput input,
                                                oklt::GnuToStdCppStageOutput& output,
                                                TranspilerSession& session)
      : _input(std::move(input)), _output(output), _session(session) {}

 protected:
  std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                        llvm::StringRef in_file) override {
    _stage = std::make_unique<SessionStage>(_session, compiler);
    if (!_stage->setUserCtx("input", &_input)) {
      // XXX error handling
      return nullptr;
    }

    return std::make_unique<GnuToCppAttrNormalizerConsumer>(*_stage);
  }

  void EndSourceFileAction() override { _output.stdCppSrc = _stage->getRewriterResult(); }

 private:
  oklt::GnuToStdCppStageInput _input;
  oklt::GnuToStdCppStageOutput& _output;
  TranspilerSession& _session;
  std::unique_ptr<SessionStage> _stage;
};

}  // namespace

namespace oklt {
tl::expected<GnuToStdCppStageOutput, int> convertGnuToStdCppAttribute(GnuToStdCppStageInput input,
                                                                      TranspilerSession& session) {
  // TODO error handling
  Twine tool_name = "okl-transpiler-normalization-to-cxx";
  Twine file_name("gnu-kernel-to-cxx.cpp");
  std::vector<std::string> args = {"-std=c++17", "-fparse-all-comments", "-I."};

  GnuToStdCppStageOutput output;
  auto input_file = std::move(input.gnuCppSrc);

  tooling::runToolOnCodeWithArgs(
      std::make_unique<GnuToStdCppAttributeNormalizerAction>(std::move(input), output, session),
      input_file, args, file_name, tool_name);

  return output;
}
}  // namespace oklt
