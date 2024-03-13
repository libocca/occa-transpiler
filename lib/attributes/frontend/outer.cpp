#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/loop.h"

#include <oklt/util/string_utils.h>

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling OUTER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "outer"},
    {ParsedAttr::AS_CXX11, OUTER_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_outer"}};

struct OuterAttribute : public ParsedAttrInfo {
    OuterAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = OUTER_ATTRIBUTE_SPELLINGS;
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

ParseResult parseOuterAttrParams(const clang::Attr& attr,
                                 OKLParsedAttr& data,
                                 SessionStage& stage) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@outer] does not take kwargs"});
    }

    if (data.args.size() > 1) {
        return tl::make_unexpected(Error{{}, "[@outer] takes at most one index"});
    }

    AttributedLoop ret{
        .type = LoopType::Outer,
        .axis = Axis::Auto,
    };

    if (auto dimSize = data.get<int>(0); dimSize.has_value()) {
        if (dimSize.value() < 0 || dimSize.value() > 2) {
            return tl::make_unexpected(Error{{}, "[@outer] argument must be 0, 1, or 2"});
        }
        ret.axis = static_cast<Axis>(dimSize.value());
    }

    return ret;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<OuterAttribute>(OUTER_ATTR_NAME,
                                                                      parseOuterAttrParams);
}
}  // namespace
