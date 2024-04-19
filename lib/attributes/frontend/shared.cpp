#include "attributes/attribute_names.h"
#include "core/handler_manager/parse_handler.h"
#include "core/transpiler_session/attributed_type_map.h"
#include "core/transpiler_session/session_stage.h"

#include "attributes/utils/parser.h"
#include "params/empty_params.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling SHARED_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, SHARED_ATTR_NAME},
    {ParsedAttr::AS_GNU, SHARED_ATTR_NAME}};

struct SharedAttribute : public ParsedAttrInfo {
    SharedAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = SHARED_ATTRIBUTE_SPELLINGS;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: this attribute appertains to functions only.
        if (!isa<VarDecl, TypeDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "variable or type declaration";
            return false;
        }

        // INFO: if VarDecl, check if array
        if (auto* var_decl = dyn_cast_or_null<VarDecl>(decl)) {
            if (!dyn_cast_or_null<ConstantArrayType>(var_decl->getType().getTypePtr())) {
                sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                    << attr << attr.isDeclspecAttribute() << "array declaration";
            }
        }
        return true;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "variable or type declaration";
        return false;
    }

    AttrHandling handleDeclAttribute(clang::Sema& sema,
                                     clang::Decl* decl,
                                     const clang::ParsedAttr& attr) const override {
        auto* stage = getStageFromASTContext(sema.Context);
        if (!stage) {
            return AttributeNotApplied;
        }

        StringRef name;
        if (!sema.checkStringLiteralArgumentAttr(attr, 0, name)) {
            return AttributeNotApplied;
        }

        llvm::SmallVector<Expr*, 4> args;
        args.reserve(attr.getNumArgs() - 1);
        for (unsigned i = 1; i < attr.getNumArgs(); i++) {
            assert(!attr.isArgIdent(i));
            args.push_back(attr.getArgAsExpr(i));
        }

        // TODO: Move attributed type registering to util, since dublication with other types
        auto* ctxAttr = AnnotateAttr::Create(sema.Context, name, args.data(), args.size(), attr);
        decl->addAttr(ctxAttr);

        auto& attrTypeMap = stage->tryEmplaceUserCtx<AttributedTypeMap>();

        // Apply Attr to Type
        // ParmVarDecl, VarDecl, FieldDecl, etc.
        if (auto val = dyn_cast<ValueDecl>(decl)) {
            QualType origType = val->getType();
            QualType modifiedType = sema.Context.getTypeOfType(origType, TypeOfKind::Qualified);

            QualType newType =
                sema.Context.getAttributedType(attr::AnnotateType, modifiedType, origType);
            val->setType(newType);

            attrTypeMap.add(newType, ctxAttr);
            return AttributeApplied;
        }

        // TypedefDecl
        if (auto typ = dyn_cast<TypeDecl>(decl)) {
            QualType origType = sema.Context.getTypeDeclType(typ);
            QualType newType =
                sema.Context.getAttributedType(attr::AnnotateType, origType, origType);
            typ->setTypeForDecl(newType.getTypePtr());

            attrTypeMap.add(newType, ctxAttr);
            return AttributeApplied;
        }

        return AttributeNotApplied;
    }
};

HandleResult parseSharedAttrParams(SessionStage& stage,
                                   const clang::Attr& attr,
                                  OKLParsedAttr& data) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@shared] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    HandlerManager::registerAttrFrontend<SharedAttribute>(SHARED_ATTR_NAME, parseSharedAttrParams);
}
}  // namespace
