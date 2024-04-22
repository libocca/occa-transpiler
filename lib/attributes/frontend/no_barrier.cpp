#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/frontend/params/empty_params.h"

#include "core/attribute_manager/attribute_manager.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling NOBARRIER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, NO_BARRIER_ATTR_NAME},
    {ParsedAttr::AS_GNU, NO_BARRIER_ATTR_NAME }};

struct NoBarrierAttribute : public ParsedAttrInfo {
    NoBarrierAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = NOBARRIER_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Suppress;
        IsStmt = true;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        if (!isa<ForStmt>(stmt)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "for statement";
            return false;
        }
        return true;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: fail for all decls
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "empty statements";
        return false;
    }
};

ParseResult parseNoBarrierAttrParams(SessionStage& stage,
                                     const clang::Attr& attr,
                                     OKLParsedAttr& data) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@nobarrier] does not take kwargs"});
    }
    if (!data.args.empty()) {
        return tl::make_unexpected(Error{{}, "[@nobarrier] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
AttributeManager::instance().registerAttrFrontend<NoBarrierAttribute>(NO_BARRIER_ATTR_NAME,
                                                                          parseNoBarrierAttrParams);
}
}  // namespace
