#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/empty_params.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling SHARED_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "shared"},
    {ParsedAttr::AS_CXX11, SHARED_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_shared"}};

struct SharedAttribute : public ParsedAttrInfoBase {
    SharedAttribute() {
        Spellings = SHARED_ATTRIBUTE_SPELLINGS;
        NumArgs = 1;
        OptArgs = 0;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Decl& decl) const override {
        // INFO: this attribute appertains to functions only.
        if (!isa<VarDecl, TypeDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "variable or type declaration";
            return false;
        }

        // INFO: if VarDecl, check if array
        if (auto* var_decl = dyn_cast_or_null<VarDecl>(&decl)) {
            if (!dyn_cast_or_null<ConstantArrayType>(var_decl->getType().getTypePtr())) {
                sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                    << attr << attr.isDeclspecAttribute() << "array declaration";
            }
        }
        return true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Stmt& stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "variable or type declaration";
        return false;
    }
};

ParseResult parseSharedAttrParams(const clang::Attr& attr,
                                  OKLParsedAttr& data,
                                  SessionStage& stage) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@shared] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<SharedAttribute>(SHARED_ATTR_NAME,
                                                                       parseSharedAttrParams);
}
}  // namespace
