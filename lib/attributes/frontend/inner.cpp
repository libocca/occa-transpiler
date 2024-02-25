#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/loop.h"

#include <oklt/util/string_utils.h>

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/ParsedAttr.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling INNER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "inner"},
    {ParsedAttr::AS_CXX11, INNER_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_inner"}};

struct InnerAttribute : public ParsedAttrInfo {
    InnerAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = INNER_ATTRIBUTE_SPELLINGS;
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
            << attr << attr.isDeclspecAttribute() << "for statement";
        return false;
    }
};

ParseResult parseInnerAttrParams(const clang::Attr& attr, SessionStage& stage) {
    auto attrData = ParseOKLAttr(attr, stage);
    if (attrData.kwargs.empty()) {
        stage.pushError(std::error_code(), "[@inner] does not take kwargs");
        return false;
    }

    if (attrData.args.size() > 1) {
        stage.pushError(std::error_code(), "[@inner] takes at most one index");
        return false;
    }

    AttributedLoop ret{
        .type = LoopType::Outer,
        .dim = Dim::Auto,
    };

    if (auto dimSize = attrData.get<int>(0); dimSize.has_value()) {
        if (dimSize.value() < 0 || dimSize.value() > 2) {
            stage.pushError(std::error_code(), "[@inner] argument must be 0, 1, or 2");
            return false;
        }
        ret.dim = static_cast<Dim>(dimSize.value());
    }

    auto ctxKey = util::pointerToStr(&attr);
    stage.tryEmplaceUserCtx<AttributedLoop>(ctxKey, ret);

    return true;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<InnerAttribute>(INNER_ATTR_NAME,
                                                                      parseInnerAttrParams);
}
}  // namespace
