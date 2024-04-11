#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/frontend/params/empty_params.h"

#include "core/attribute_manager/attribute_manager.h"
#include "core/attribute_manager/attributed_type_map.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling EXCLUSIVE_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, EXCLUSIVE_ATTR_NAME},
    {ParsedAttr::AS_GNU, EXCLUSIVE_ATTR_NAME}};

struct ExclusiveAttribute : public ParsedAttrInfo {
    ExclusiveAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = EXCLUSIVE_ATTRIBUTE_SPELLINGS;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: this attribute appertains to variable declarations only.
        if (!isa<VarDecl, TypeDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "variable or type declaration";
            return false;
        }
        return true;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "type or variable declarations";
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

        // Apply Attr to VarDecl
        if (auto val = dyn_cast<VarDecl>(decl)) {
            QualType origType = val->getType();
            QualType modifiedType = sema.Context.getTypeOfType(origType, TypeOfKind::Qualified);

            QualType newType =
                sema.Context.getAttributedType(attr::AnnotateType, modifiedType, origType);
            val->setType(newType);

            attrTypeMap.add(newType, ctxAttr);
            return AttributeApplied;
        }

        // TypeDecl
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

HandleResult parseExclusiveAttrParams(SessionStage& stage,
                                      const clang::Attr& attr,
                                      OKLParsedAttr& data) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@exclusive] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerKernelHandler() {
    AttributeManager::instance().registerAttrFrontend<ExclusiveAttribute>(EXCLUSIVE_ATTR_NAME,
                                                                          parseExclusiveAttrParams);
}
}  // namespace
