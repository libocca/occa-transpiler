#include "attributes/attribute_names.h"
#include "attributes/frontend/params/empty_params.h"
#include "attributes/utils/parser.h"

#include "core/handler_manager/parse_handler.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling ATOMIC_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, ATOMIC_ATTR_NAME},
    {ParsedAttr::AS_GNU, ATOMIC_ATTR_NAME}};

struct AtomicAttribute : public ParsedAttrInfo {
    AtomicAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = ATOMIC_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Suppress;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        if (!isa<Expr, CompoundStmt>(stmt)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "expression or compound statement";
            return false;
        }
        return true;
    }

    bool diagAppertainsToDecl(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Decl* decl) const override {
        // INFO: fail for all decls
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "expression or compound statement";
        return false;
    }
};

HandleResult parseAtomicAttrParams(SessionStage& stage, const Attr& attr, OKLParsedAttr& data) {
    if (!data.args.empty() || !data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@atomic] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    HandlerManager::instance().registerAttrFrontend<AtomicAttribute>(ATOMIC_ATTR_NAME,
                                                                     parseAtomicAttrParams);
}
}  // namespace
