#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling EXCLUSIVE_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "exclusive"},
    {ParsedAttr::AS_CXX11, EXCLUSIVE_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_exclusive"}};

struct ExclusiveAttribute : public ParsedAttrInfo {
    ExclusiveAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = EXCLUSIVE_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Annotate;
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
};

tl::expected<std::any, Error> parseExclusiveAttrParams(const clang::Attr* a, SessionStage&) {
    return true;
}

__attribute__((constructor)) void registerKernelHandler() {
    AttributeManager::instance().registerAttrFrontend<ExclusiveAttribute>(EXCLUSIVE_ATTR_NAME,
                                                                          parseExclusiveAttrParams);
}
}  // namespace
