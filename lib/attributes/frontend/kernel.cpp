#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/empty_params.h"

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling KERNEL_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, KERNEL_ATTR_NAME},
    {ParsedAttr::AS_CXX11, "kernel"},
    {ParsedAttr::AS_GNU, "okl_kernel"}};

struct KernelAttribute : public ParsedAttrInfo {
    KernelAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = KERNEL_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Annotate;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: this decl function can be saved to global map.
        //       in this case there is no need to make attribute !!!
        // INFO: this attribute appertains to functions only.
        if (!isa<FunctionDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "functions";
            return false;
        }

        auto func = dyn_cast<FunctionDecl>(decl);
        auto returnTypeStr = func->getReturnType().getAsString();
        if (returnTypeStr != "void") {
            sema.Diag(attr.getLoc(), diag::err_type_attribute_wrong_type)
                << attr << "functions with [void] return" << returnTypeStr;
            return false;
        }

        return true;
    }
};

ParseResult parseKernelAttrParams(const clang::Attr& attr, SessionStage& stage) {
    auto attrData = ParseOKLAttr(attr, stage);
    if (!attrData.args.empty() || !attrData.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@kernel] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<KernelAttribute>(KERNEL_ATTR_NAME,
                                                                       parseKernelAttrParams);
}
}  // namespace
