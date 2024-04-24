#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/frontend/params/empty_params.h"

#include "core/handler_manager/parse_handler.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling KERNEL_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, KERNEL_ATTR_NAME},
    {ParsedAttr::AS_GNU, KERNEL_ATTR_NAME}};

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
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type)
                << attr << attr.isDeclspecAttribute() << "functions";
            return false;
        }

        auto func = dyn_cast<FunctionDecl>(decl);
        auto returnTypeStr = func->getReturnType().getAsString();
        if (returnTypeStr != "void") {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type)
                << attr << ":" << "functions with [void] return types";
            return false;
        }

        return true;
    }
};

HandleResult parseKernelAttrParams(SessionStage& stage,
                                   const clang::Attr& attr,
                                   OKLParsedAttr& data) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@kernel] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerKernelAttrFrontend() {
    registerAttrFrontend<KernelAttribute>(KERNEL_ATTR_NAME, parseKernelAttrParams);
}
}  // namespace
