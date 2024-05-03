#include "attributes/frontend/params/dim.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"

#include "core/diag/diag_handler.h"
#include "core/handler_manager/parse_handler.h"
#include "core/transpiler_session/attributed_type_map.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling DIM_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, DIM_ATTR_NAME},
    {ParsedAttr::AS_GNU, DIM_ATTR_NAME}};

struct DimAttribute : public ParsedAttrInfo {
    DimAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = DIM_ATTRIBUTE_SPELLINGS;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        if (!isa<VarDecl, ParmVarDecl, TypeDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute()
                << "type or variable declarations";
            return false;
        }
        return true;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute()
            << "type or variable declarations";
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

class DimDiagHandler : public DiagHandler {
   public:
    DimDiagHandler()
        : DiagHandler(diag::err_typecheck_call_not_function){};

    bool HandleDiagnostic(SessionStage& session, DiagLevel level, const Diagnostic& info) override {
        if (info.getArgKind(0) != DiagnosticsEngine::ak_qualtype)
            return false;

        QualType qt = QualType::getFromOpaquePtr(reinterpret_cast<void*>(info.getRawArg(0)));

        static llvm::ManagedStatic<SmallVector<StringRef>> attrNames = {};
        if (attrNames->empty()) {
            for (auto v : DIM_ATTRIBUTE_SPELLINGS) {
                attrNames->push_back(v.NormalizedFullName);
            }
        };

        auto& ctx = session.getCompiler().getASTContext();
        auto& attrTypeMap = session.tryEmplaceUserCtx<AttributedTypeMap>();
        if (attrTypeMap.has(ctx, qt, *attrNames))
            return true;

        return false;
    }
};

HandleResult parseDimAttrParams(SessionStage& stage, const clang::Attr& attr, OKLParsedAttr& data) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@dim] does not take kwargs"});
    }

    if (data.args.empty()) {
        return tl::make_unexpected(Error{{}, "[@dim] expects at least one argument"});
    }

    AttributedDim ret;
    ret.dim.reserve(data.args.size());
    for (auto arg : data.args) {
        ret.dim.emplace_back(arg.getRaw());
    }

    return ret;
}

__attribute__((constructor)) void registerDimAttrFrontend() {
    registerAttrFrontend<DimAttribute>(DIM_ATTR_NAME, parseDimAttrParams);
    // for suppression of func call error that potentially is dim calls
    static DiagHandlerRegistry::Add<DimDiagHandler> diag_dim("DimDiagHandler", "");
}
}  // namespace
