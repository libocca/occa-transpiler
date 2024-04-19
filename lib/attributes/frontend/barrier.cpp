#include "attributes/frontend/params/barrier.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"

#include "core/handler_manager/parse_handler.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/Sema.h>

namespace {
using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling BARRIER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, BARRIER_ATTR_NAME},
    {ParsedAttr::AS_GNU, BARRIER_ATTR_NAME}};

struct BarrierAttribute : public ParsedAttrInfo {
    BarrierAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = BARRIER_ATTRIBUTE_SPELLINGS;
        AttrKind = clang::AttributeCommonInfo::AT_Suppress;
        IsStmt = true;
    }

    bool diagAppertainsToStmt(clang::Sema& sema,
                              const clang::ParsedAttr& attr,
                              const clang::Stmt* stmt) const override {
        if (!isa<NullStmt>(stmt)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "empty statement";
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

HandleResult parseBarrierAttrParams(SessionStage& stage,
                                    const clang::Attr& attr,
                                    OKLParsedAttr& data) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@barrier] does not take kwargs"});
    }

    AttributedBarrier ret{
        .type = BarrierType::syncDefault,
    };

    if (data.args.size() > 1) {
        return tl::make_unexpected(Error{{}, "[@barrier] takes at most one argument"});
    }

    if (!data.args.empty()) {
        auto firstParam = data.get<std::string>(0);
        if (!firstParam.has_value()) {
            return tl::make_unexpected(
                Error{{}, "[@barrier] must have no arguments or have one string argument"});
        }

        if (firstParam.value() != "warp") {
            return tl::make_unexpected(
                Error{{},
                      util::fmt("[@barrier] has an invalid barrier type: {}", firstParam.value())
                          .value()});
        }

        ret.type = BarrierType::syncWarp;
    }

    return ret;
}

__attribute__((constructor)) void registerAttrFrontend() {
    HandlerManager::registerAttrFrontend<BarrierAttribute>(BARRIER_ATTR_NAME,
                                                           parseBarrierAttrParams);
}
}  // namespace
