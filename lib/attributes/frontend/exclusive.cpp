#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "params/empty_params.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling EXCLUSIVE_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "exclusive"},
    {ParsedAttr::AS_CXX11, EXCLUSIVE_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_exclusive"}};

struct ExclusiveAttribute : public ParsedAttrInfoBase {
    ExclusiveAttribute() {
        Spellings = EXCLUSIVE_ATTRIBUTE_SPELLINGS;
        NumArgs = 1;
        OptArgs = 0;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Decl& decl) const override {
        // INFO: this attribute appertains to variable declarations only.
        if (!isa<VarDecl, TypeDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "variable or type declaration";
            return false;
        }
        return true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Stmt& stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "type or variable declarations";
        return false;
    }
};

ParseResult parseExclusiveAttrParams(const clang::Attr& attr,
                                     OKLParsedAttr& data,
                                     SessionStage& stage) {
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
