#include "attributes/attribute_names.h"
#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"

#include "core/handler_manager/parse_handler.h"
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
        AttrKind = clang::AttributeCommonInfo::AT_Annotate;
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
};

HandleResult parseRestrictAttrParams(SessionStage& stage,
                                     const clang::Attr& attr,
                                    OKLParsedAttr& data) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@atomic] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    HandlerManager::registerAttrFrontend<RestrictAttribute>(RESTRICT_ATTR_NAME,
                                                            parseRestrictAttrParams);
}
}  // namespace
