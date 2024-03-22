#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "params/empty_params.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling RESTRICT_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "restrict"},
    {ParsedAttr::AS_CXX11, RESTRICT_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_restrict"}};

struct RestrictAttribute : public ParsedAttrInfoBase {
    RestrictAttribute() {
        Spellings = RESTRICT_ATTRIBUTE_SPELLINGS;
        NumArgs = 1;
        OptArgs = 0;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Decl& decl) const override {
        // INFO: this can to applied to following decl
        if (!isa<VarDecl, ParmVarDecl, TypeDecl, FieldDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << ": can be applied only for parameters of pointer type in function";
            return false;
        }
        const auto type = [&sema](const clang::Decl* decl) {
            switch (decl->getKind()) {
                case Decl::Field:
                    return cast<FieldDecl>(decl)->getType();
                case Decl::Typedef:
                    return sema.Context.getTypeDeclType(dyn_cast<TypeDecl>(decl));
                default:
                    return cast<VarDecl>(decl)->getType();
            }
        }(&decl);

        if (!type->isPointerType() && !type->isArrayType()) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << ": supports only pointer type";
            return false;
        }
        return true;
    }
};

ParseResult parseRestrictAttrParams(const clang::Attr& attr,
                                    OKLParsedAttr& data,
                                    SessionStage& stage) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@atomic] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<RestrictAttribute>(RESTRICT_ATTR_NAME,
                                                                         parseRestrictAttrParams);
}
}  // namespace
