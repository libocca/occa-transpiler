#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "attributes/frontend/params/loop.h"

#include "core/handler_manager/parse_handler.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling INNER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, INNER_ATTR_NAME},
    {ParsedAttr::AS_GNU, INNER_ATTR_NAME}};

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

HandleResult parseInnerAttrParams(SessionStage& stage,
                                  const clang::Attr& attr,
                                  OKLParsedAttr& data) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@inner] does not take kwargs"});
    }

    if (data.args.size() > 1) {
        return tl::make_unexpected(Error{{}, "[@inner] takes at most one index"});
    }

    AttributedLoop ret{
        .type = LoopType::Inner,
        .axis = Axis::Auto,
    };

    if (auto dimSize = data.get<int>(0); dimSize.has_value()) {
        if (dimSize.value() < 0 || dimSize.value() > 2) {
            return tl::make_unexpected(Error{{}, "[@inner] argument must be 0, 1, or 2"});
        }
        ret.axis = static_cast<Axis>(dimSize.value());
    }

    return ret;
}

__attribute__((constructor)) void registerAttrFrontend() {
    HandlerManager::instance().registerAttrFrontend<InnerAttribute>(INNER_ATTR_NAME,
                                                                    parseInnerAttrParams);
}
}  // namespace
