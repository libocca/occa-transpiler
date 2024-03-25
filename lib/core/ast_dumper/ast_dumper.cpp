#include "core/ast_dumper/ast_dumper.h"
#include "core/attribute_manager/attribute_store.h"
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"
#include "core/utils/attributes.h"

#include <clang/AST/ASTContext.h>
#include <clang/AST/ASTNodeTraverser.h>
#include <clang/AST/Comment.h>
#include <clang/AST/TextNodeDumper.h>

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/raw_ostream.h>

namespace {
using namespace oklt;
using namespace clang;

class ASTDumper : public ASTNodeTraverser<ASTDumper, TextNodeDumper> {
    using Base = ASTNodeTraverser<ASTDumper, TextNodeDumper>;

   public:
    ASTDumper(raw_ostream& os, const ASTContext& ctx, AttributeStore& attrs, bool enableColors)
        : _nodeDumper(os, ctx, enableColors),
          _os(os),
          _attrs(attrs),
          _colors(enableColors) {}

    TextNodeDumper& doGetNodeDelegate() { return _nodeDumper; }

    // NOTE: Original ASTDumper behaves differently for FunctionTemplateDecl, ClassTemplateDecl and
    // VarTemplateDecl. Therefore, we have to be similar.

    template <typename SpecializationDecl>
    void dumpTemplateDeclSpecialization(const SpecializationDecl* D,
                                        bool dumpExplicitInst,
                                        bool dumpRefOnly) {
        bool dumpedAny = false;
        for (const auto* redeclWithBadType : D->redecls()) {
            auto* redecl = cast<SpecializationDecl>(redeclWithBadType);
            switch (redecl->getTemplateSpecializationKind()) {
                case TSK_ExplicitInstantiationDeclaration:
                case TSK_ExplicitInstantiationDefinition:
                    if (!dumpExplicitInst) {
                        break;
                    }
                    [[fallthrough]];
                case TSK_Undeclared:
                case TSK_ImplicitInstantiation:
                    if (dumpRefOnly) {
                        _nodeDumper.dumpDeclRef(redecl);
                    } else {
                        Visit(redecl);
                    }
                    dumpedAny = true;
                    break;
                case TSK_ExplicitSpecialization:
                    break;
            }
        }

        if (!dumpedAny) {
            _nodeDumper.dumpDeclRef(D);
        }
    }

    template <typename TemplateDecl>
    void dumpTemplateDecl(const TemplateDecl* D, bool dumpExplicitInst) {
        dumpTemplateParameters(D->getTemplateParameters());

        Base::Visit(D->getTemplatedDecl());

        if (GetTraversalKind() == TK_AsIs) {
            for (const auto* Child : D->specializations()) {
                dumpTemplateDeclSpecialization(Child, dumpExplicitInst, !D->isCanonicalDecl());
            }
        }
    }

    // NOTE: We are unable to overload ASTNodeTraverser::Visit, therefore we will overload
    // all possible Decl and Stmt.

#define PTR(CLASS) typename llvm::make_const_ptr<CLASS>::type
#define DECL(DERIVED, BASE)                                                            \
    void Visit##DERIVED##Decl(PTR(DERIVED##Decl) D) {                                  \
        if constexpr (std::is_same_v<PTR(FunctionTemplateDecl), PTR(DERIVED##Decl)> || \
                      std::is_same_v<PTR(ClassTemplateDecl), PTR(DERIVED##Decl)> ||    \
                      std::is_same_v<PTR(VarTemplateDecl), PTR(DERIVED##Decl)>) {      \
            dumpTemplateDecl(D, false);                                                \
        }                                                                              \
        if (D) {                                                                       \
            for (const auto& A : _attrs.get(*D)) {                                     \
                Base::Visit(A);                                                        \
            }                                                                          \
        }                                                                              \
        Base::Visit##DERIVED##Decl(D);                                                 \
    }
#include <clang/AST/DeclNodes.inc>

#define STMT(CLASS, PARENT)                        \
    void Visit##CLASS(PTR(CLASS) S) {              \
        if (S) {                                   \
            for (const auto& A : _attrs.get(*S)) { \
                Base::Visit(A);                    \
            }                                      \
        }                                          \
        Base::Visit##CLASS(S);                     \
    }
#include <clang/AST/StmtNodes.inc>

    void VisitAttributedType(const AttributedType* T) {
        for (const auto& A : _attrs.get(*T)) {
            Base::Visit(A);
        }

        if (T->getModifiedType() != T->getEquivalentType()) {
            Base::Visit(T->getModifiedType());
        }
    }

    void VisitAnnotateAttr(const AnnotateAttr* A) {
        Base::VisitAnnotateAttr(A);
        if (A && isOklAttribute(*A)) {
            _os << " " << A->getNormalizedFullName();
        }
    }
    void VisitAnnotateTypeAttr(const AnnotateTypeAttr* A) {
        Base::VisitAnnotateTypeAttr(A);
        if (A && isOklAttribute(*A)) {
            _os << " " << A->getNormalizedFullName();
        }
    }
    void VisitSuppressAttr(const SuppressAttr* A) {
        Base::VisitSuppressAttr(A);
        if (A && isOklAttribute(*A)) {
            _os << " " << A->getNormalizedFullName();
        }
    }

   private:
    TextNodeDumper _nodeDumper;
    raw_ostream& _os;
    AttributeStore& _attrs;
    const bool _colors;
};

}  // namespace

namespace oklt {
using namespace clang;

static std::pair<ASTContext&, AttributeStore&> getASTContext(SessionStage& stage) {
    auto& ctx = stage.getCompiler().getASTContext();
    auto& attrStore = stage.tryEmplaceUserCtx<AttributeStore>(ctx);
    return {ctx, attrStore};
}

LLVM_DUMP_METHOD void dump(Decl* D, SessionStage& stage, raw_ostream& os) {
    auto [ctx, attrs] = getASTContext(stage);
    ASTDumper dumper(os, ctx, attrs, ctx.getDiagnostics().getShowColors());
    dumper.Visit(D);
}

LLVM_DUMP_METHOD void dump(Stmt* S, SessionStage& stage, raw_ostream& os) {
    auto [ctx, attrs] = getASTContext(stage);
    ASTDumper dumper(os, ctx, attrs, ctx.getDiagnostics().getShowColors());
    dumper.Visit(S);
}

LLVM_DUMP_METHOD void dump(QualType qt, SessionStage& stage, llvm::raw_ostream& os) {
    auto [ctx, attrs] = getASTContext(stage);
    ASTDumper dumper(os, ctx, attrs, ctx.getDiagnostics().getShowColors());
    dumper.Visit(qt);
}

LLVM_DUMP_METHOD void dump(Type* t, SessionStage& stage, llvm::raw_ostream& os) {
    dump(QualType(t, 0), stage, os);
}

LLVM_DUMP_METHOD void dump(APValue* ap, SessionStage& stage, raw_ostream& os) {
    auto [ctx, attrs] = getASTContext(stage);
    ASTDumper dumper(os, ctx, attrs, ctx.getDiagnostics().getShowColors());
    dumper.Visit(*ap, ctx.getPointerType(ctx.CharTy));
}

}  // namespace oklt
