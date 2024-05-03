#include <oklt/core/error.h>

#include "core/diag/diag_consumer.h"
#include "core/transpiler_session/session_stage.h"
#include "core/utils/attributes.h"

#include "pipeline/core/stage_action_names.h"
#include "pipeline/core/stage_action_registry.h"
#include "pipeline/utils/okl_attribute.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/RecursiveASTVisitor.h>
#include <clang/Frontend/CompilerInstance.h>
#include <clang/Tooling/Tooling.h>

#include <spdlog/spdlog.h>

namespace {

using namespace oklt;
using namespace clang;

void removeAttr(oklt::Rewriter& rewriter, const Attr& attr) {
    auto arange = getAttrFullSourceRange(attr);
    rewriter.RemoveText(arange);
}

std::string getArgAsStr(const SuppressAttr& attr) {
    if (attr.diagnosticIdentifiers_size() == 0) {
        return "";
    }
    return attr.diagnosticIdentifiers_begin()[0].str();
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

    auto& sm = stage.getCompiler().getSourceManager();
    auto& mapper = stage.getSession().getOriginalSourceMapper();

    for (auto* attr : e.getAttrs()) {
        auto attrBegLoc = attr->getRange().getBegin();
        auto prevFidAttrOffset = sm.getDecomposedLoc(attrBegLoc);

        if (attr->isCXX11Attribute()) {
            mapper.updateAttributeOffset(prevFidAttrOffset, attrBegLoc, stage.getRewriter());
            continue;
        }

        if (!oklt::isOklAttribute(*attr)) {
            continue;
        }

        if ((*lastProccesedAttr) && ((*lastProccesedAttr)->getLoc() == attr->getLoc())) {
            continue;
        }

        const auto* targetAttr = dyn_cast_or_null<AttrType>(attr);
        if (!targetAttr) {
            continue;
        }

        removeAttr(stage.getRewriter(), *attr);
        auto newAttrLoc = e.getBeginLoc();

        insertNormalizedAttr(e, *targetAttr, stage);
        *lastProccesedAttr = attr;

        // Add offset, since after removal of GNU, it will point at the beginning of attribute
        mapper.updateAttributeOffset(
            prevFidAttrOffset, newAttrLoc, stage.getRewriter(), CXX_ATTRIBUTE_BEGIN_TO_NAME_OFFSET);
    }

    return true;
}

// Traverse AST and convert GNU attributes and to standard C++ attribute
// syntax and unified source location
class GnuToStdAttrNormalizerVisitor : public RecursiveASTVisitor<GnuToStdAttrNormalizerVisitor> {
   public:
    explicit GnuToStdAttrNormalizerVisitor(SessionStage& stage)
        : _stage(stage),
          _sm(stage.getCompiler().getSourceManager()) {}

    bool TraverseAttributedStmt(AttributedStmt* as) {
        if (!as) {
            SPDLOG_WARN("attributed statement is nullptr");
            return false;
        }

        if (_sm.isInSystemHeader(as->getEndLoc())) {
            return true;
        }

        if (!tryToNormalizeAttrExpr<SuppressAttr>(*as, _stage, &_lastProccesedAttr)) {
            return false;
        }

        return RecursiveASTVisitor<GnuToStdAttrNormalizerVisitor>::TraverseAttributedStmt(as);
    }

    bool TraverseDecl(clang::Decl* d) {
        if (!d) {
            SPDLOG_WARN("declaration is nullptr");
            return false;
        }

        if (_sm.isInSystemHeader(d->getLocation())) {
            return true;
        }

        if (!d->hasAttrs()) {
            return RecursiveASTVisitor<GnuToStdAttrNormalizerVisitor>::TraverseDecl(d);
        }
        auto ok = tryToNormalizeAttrExpr<AnnotateAttr>(*d, _stage, &_lastProccesedAttr);
        if (!ok) {
            return false;
        }

        return RecursiveASTVisitor<GnuToStdAttrNormalizerVisitor>::TraverseDecl(d);
    }

   private:
    const Attr* _lastProccesedAttr{nullptr};
    SessionStage& _stage;
    const SourceManager& _sm;
};

// ASTConsumer to run GNU to C++ attribute replacing
class GnuToStdAttrNormalizerConsumer : public ASTConsumer {
   public:
    explicit GnuToStdAttrNormalizerConsumer(SessionStage& stage)
        : _stage(stage),
          _normalizer_visitor(_stage) {}

    // Override the method that gets called for each parsed top-level
    // declaration.
    void HandleTranslationUnit(ASTContext& ctx) override {
        if (spdlog::get_level() == spdlog::level::trace) {
            ctx.getTranslationUnitDecl()->dump(llvm::outs());
        }
        TranslationUnitDecl* decl = ctx.getTranslationUnitDecl();
        _normalizer_visitor.TraverseDecl(decl);
    }

   private:
    SessionStage& _stage;
    GnuToStdAttrNormalizerVisitor _normalizer_visitor;
};

class GnuToStdAttrNormalizer : public StageAction {
   public:
    GnuToStdAttrNormalizer() { _name = GNU_TO_STD_ATTR_NORMALIZER_STAGE; }

   protected:
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                          llvm::StringRef in_file) override {
        compiler.getDiagnostics().setClient(new DiagConsumer(*_stage));
        return std::make_unique<GnuToStdAttrNormalizerConsumer>(*_stage);
    }

    RewriterProxyType getRewriterType() const override { return RewriterProxyType::WithDeltaTree; }
};

StagePluginRegistry::Add<GnuToStdAttrNormalizer> gnuToCppAttrNormalizer(
    GNU_TO_STD_ATTR_NORMALIZER_STAGE,
    "");
}  // namespace
