#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "params/dim.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling DIMORDER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "dimOrder"},
    {ParsedAttr::AS_CXX11, DIMORDER_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_dimOrder"}};

struct DimOrderAttribute : public ParsedAttrInfoBase {
    DimOrderAttribute() {
        // AttrKind = AttributeKind::MODIFIER;
        Spellings = DIMORDER_ATTRIBUTE_SPELLINGS;
        NumArgs = 1;
        OptArgs = 0;
        IsType = 1;
        HasCustomParsing = 1;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Decl& decl) const override {
        if (!isa<VarDecl, ParmVarDecl, TypedefDecl, FieldDecl>(decl)) {
            sema.Diag(attr.getLoc(), diag::warn_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute()
                << "type, struct/union/class field or variable declarations";
            return false;
        }
        return true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Stmt& stmt) const override {
        // INFO: fail for all statements
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute()
            << "type, struct/union/class field or variable declarations";
        return false;
    }
};

ParseResult parseDimOrderAttrParams(const clang::Attr& attr,
                                    OKLParsedAttr& data,
                                    SessionStage& stage) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@dimOrder] does not take kwargs"});
    }

    if (data.args.empty()) {
        return tl::make_unexpected(Error{{}, "[@dimOrder] expects at least one argument"});
    }

    AttributedDimOrder ret;
    for (auto arg : data.args) {
        auto idx = arg.get<size_t>();
        if (!idx.has_value()) {
            return tl::make_unexpected(
                Error{{}, "[@dimOrder] expects expects positive integer index"});
        }

        auto it = std::find(ret.idx.begin(), ret.idx.end(), idx.value());
        if (it != ret.idx.end()) {
            return tl::make_unexpected(Error{{}, "[@dimOrder] Duplicate index"});
        }

        ret.idx.push_back(idx.value());
    }

    return ret;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<DimOrderAttribute>(DIMORDER_ATTR_NAME,
                                                                         parseDimOrderAttrParams);
}
}  // namespace
