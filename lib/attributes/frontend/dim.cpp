#include "oklt/core/attribute_names.h"
#include "oklt/core/diag/diag_consumer.h"
#include "oklt/core/diag/diag_handler.h"
#include "oklt/core/transpiler_session/transpiler_session.h"
#include "oklt/core/attribute_manager/attributed_type_map.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling DIM_ATTRIBUTE_SPELLINGS[] = {
  {ParsedAttr::AS_CXX11, "dim"},
  {ParsedAttr::AS_CXX11, DIM_ATTR_NAME},
  {ParsedAttr::AS_GNU, "okl_dim"}};

struct DimAttribute : public ParsedAttrInfo {
  DimAttribute() {
    NumArgs = 1;
    OptArgs = 6;
    Spellings = DIM_ATTRIBUTE_SPELLINGS;
    IsType = 1;
    HasCustomParsing = 1;
  }

  bool diagAppertainsToDecl(clang::Sema& sema,
                            const clang::ParsedAttr& attr,
                            const clang::Decl* decl) const override {
    if (!isa<VarDecl, ParmVarDecl, TypedefDecl, FieldDecl>(decl)) {
      sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
        << attr << attr.isDeclspecAttribute() << "typedefs or variable declarations";
      return false;
    }
    return true;
  }

  AttrHandling handleDeclAttribute(clang::Sema &sema, clang::Decl *decl, const clang::ParsedAttr &attr) const override {
    auto *stage = getStageFromASTContext(sema.Context);
    if (!stage) {
      return AttributeNotApplied;
    }

    StringRef name;
    if (!sema.checkStringLiteralArgumentAttr(attr, 0, name)) {
      return AttributeNotApplied;
    }

    llvm::SmallVector<Expr *, 4> args;
    args.reserve(attr.getNumArgs() - 1);
    for (unsigned i = 1; i < attr.getNumArgs(); i++) {
      assert(!attr.isArgIdent(i));
      args.push_back(attr.getArgAsExpr(i));
    }

    auto *ctxAttr = AnnotateAttr::Create(sema.Context, name, args.data(), args.size(), attr);
    decl->addAttr(ctxAttr);

    // ValueDecl:
    //   ParmVarDecl -- func param
    //   VarDecl -- var
    //   FieldDecl -- struct field
    // TypeDecl:
    //   TypedefDecl -- typedef

    auto & attrTypeMap = stage->tryEmplaceUserCtx<AttributedTypeMap>();

    // Apply Attr to Type
    // ParmVarDecl, VarDecl, FieldDecl, etc.
    if (auto val = dyn_cast<ValueDecl>(decl)) {
      QualType origType = val->getType();
      QualType newType = sema.Context.getAttributedType(attr::AnnotateType, origType, origType);
      val->setType(newType);

      attrTypeMap.add(newType, ctxAttr);
      return AttributeApplied;
    }

    // TypedefDecl
    if (auto typ = dyn_cast<TypeDecl>(decl)) {
      QualType origType = sema.Context.getTypeDeclType(typ);
      QualType newType = sema.Context.getAttributedType(attr::AnnotateType, origType, origType);
      typ->setTypeForDecl(newType.getTypePtr());

      attrTypeMap.add(newType, ctxAttr);
      return AttributeApplied;
    }

    return AttributeNotApplied;
  }
};

class DimDiagHandler : public DiagHandler {
 public:
  DimDiagHandler() : DiagHandler(diag::err_typecheck_call_not_function){};

  bool HandleDiagnostic(SessionStage& session, DiagLevel level, const Diagnostic& info) override {
    if (info.getArgKind(0) != DiagnosticsEngine::ak_qualtype)
      return false;

    QualType qt = QualType::getFromOpaquePtr(reinterpret_cast<void*>(info.getRawArg(0)));
    if (auto aqt = dyn_cast_or_null<ArrayType>(qt)) {
      qt = aqt->getElementType();
    }

    static llvm::ManagedStatic<SmallVector<StringRef>> attrNames = {};
    if (attrNames->empty()) {
      for (auto v : DIM_ATTRIBUTE_SPELLINGS) {
        attrNames->push_back(v.NormalizedFullName);
      }
    };

    auto & ctx = session.getCompiler().getASTContext();
    auto & attrTypeMap = session.tryEmplaceUserCtx<AttributedTypeMap>();
    if (attrTypeMap.has(ctx, qt, *attrNames))
      return true;

    return false;
  }
};

ParsedAttrInfoRegistry::Add<DimAttribute> register_okl_sim(DIM_ATTR_NAME, "");
oklt::DiagHandlerRegistry::Add<DimDiagHandler> diag_dim("DimDiagHandler", "");

} // namespace
