#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/empty_params.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling RESTRICT_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "restrict"},
    {ParsedAttr::AS_CXX11, RESTRICT_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_restrict"}};

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
        // INFO: this can be as the function argument
        if (!isa<ParmVarDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << ": can be applied only for parameters of pointer type in function";
            return false;
        }
        const auto* param = cast<ParmVarDecl>(decl);
        if (!param->getType()->isPointerType()) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << ": supports only pointer type";
            return false;
        }
        return true;
    }
};

ParseResult parseRestrictAttrParams(const clang::Attr& attr, SessionStage& stage) {
    auto attrData = ParseOKLAttr(attr, stage);
    if (!attrData.args.empty() || !attrData.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@atomic] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<RestrictAttribute>(RESTRICT_ATTR_NAME,
                                                                         parseRestrictAttrParams);
}
}  // namespace
