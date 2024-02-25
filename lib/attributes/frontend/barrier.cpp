#include "attributes/attribute_names.h"
#include "core/attribute_manager/attribute_manager.h"

#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/barrier.h"

#include <oklt/util/string_utils.h>

#include "clang/Basic/DiagnosticSema.h"
#include "clang/Sema/Sema.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling BARRIER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "barrier"},
    {ParsedAttr::AS_CXX11, BARRIER_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_barrier"}};

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

ParseResult parseBarrierAttrParams(const clang::Attr& attr, SessionStage& stage) {
    auto attrData = ParseOKLAttr(attr, stage);
    if (attrData.kwargs.empty()) {
        stage.pushError(std::error_code(), "[@barrier] does not take kwargs");
        return false;
    }

    AttributedBarrier ret{
        .type = BarrierType::syncDefault,
    };

    if (attrData.args.size() > 1) {
        stage.pushError(std::error_code(), "[@barrier] takes at most one argument");
        return false;
    }

    if (!attrData.args.empty()) {
        auto firstParam = attrData.get<std::string>(0);
        if (!firstParam.has_value()) {
            stage.pushError(std::error_code(),
                            "[@barrier] must have no arguments or have one string argument");
            return false;
        }

        if (firstParam.value() != "warp") {
            stage.pushError(std::error_code(),
                            "[@barrier] has an invalid barrier type: " + firstParam.value());
            return false;
        }

        ret.type = BarrierType::syncWarp;
    }

    auto ctxKey = util::pointerToStr(static_cast<const void*>(&attr));
    stage.tryEmplaceUserCtx<AttributedBarrier>(ctxKey, ret);

    return true;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<BarrierAttribute>(BARRIER_ATTR_NAME,
                                                                        parseBarrierAttrParams);
}
}  // namespace
