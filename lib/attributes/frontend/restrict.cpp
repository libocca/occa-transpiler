#include "attributes/attribute_names.h"
#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"

#include "core/handler_manager/parse_handler.h"
#include "core/transpiler_session/attributed_type_map.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling RESTRICT_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, RESTRICT_ATTR_NAME},
    {ParsedAttr::AS_GNU, RESTRICT_ATTR_NAME}};

struct RestrictAttribute : public ParsedAttrInfo {
    RestrictAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = RESTRICT_ATTRIBUTE_SPELLINGS;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: this can to applied to following decl
        if (!isa<VarDecl, ParmVarDecl, TypeDecl, FieldDecl, FunctionDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << ":"
                << "parameters of pointer type in function";
            return false;
        }
        const auto type = [&sema](const clang::Decl* decl) {
            switch (decl->getKind()) {
                case Decl::Field:
                    return cast<FieldDecl>(decl)->getType();
                case Decl::Typedef:
                    return sema.Context.getTypeDeclType(dyn_cast<TypeDecl>(decl));
                case Decl::Function:
                    return cast<FunctionDecl>(decl)->getReturnType();
                default:
                    return cast<VarDecl>(decl)->getType();
            }
        }(decl);

        if (!type->isPointerType() && !type->isArrayType()) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << ":" << "pointer type";
            return false;
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

HandleResult parseRestrictAttrParams(SessionStage& stage,
                                     const clang::Attr& attr,
                                    OKLParsedAttr& data) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@atomic] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerRestrictAttrFrontend() {
    registerAttrFrontend<RestrictAttribute>(RESTRICT_ATTR_NAME, parseRestrictAttrParams);
}
}  // namespace
