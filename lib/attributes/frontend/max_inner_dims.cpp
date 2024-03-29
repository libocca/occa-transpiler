#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "params/loop.h"

namespace {
using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling MAX_INNER_DIMS_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "max_inner_dims"},
    {ParsedAttr::AS_CXX11, MAX_INNER_DIMS},
    {ParsedAttr::AS_GNU, "okl_max_inner_dims"}};

struct MaxInnerDims : public ParsedAttrInfoBase {
    MaxInnerDims() {
        AttrKind = AttributeKind::MODIFIER;
        Spellings = MAX_INNER_DIMS_ATTRIBUTE_SPELLINGS;
        NumArgs = 1;
        OptArgs = 0;
        IsStmt = true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Stmt& stmt) const override {
        if (!isa<ForStmt>(stmt)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "for statement";
            return false;
        }
        return true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Decl& decl) const override {
        // INFO: fail for all decls
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "for statement";
        return false;
    }
};

ParseResult parseMaxInnerDims(const clang::Attr& attr, OKLParsedAttr& data, SessionStage& stage) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@max_inner_dims] does not take kwargs"});
    }

    if (data.args.empty()) {
        return tl::make_unexpected(Error{{}, "[@max_inner_dims] expects at least one argument"});
    }

    if (data.args.size() > 3) {
        return tl::make_unexpected(Error{{}, "[@max_inner_dims] takes at most 3 arguments"});
    }

    AttributedLoopInnerSize ret{};
    for (auto i = size_t(0); i < data.args.size(); ++i) {
        auto dimSize = data.get<int>(i);
        if (!dimSize.has_value() || dimSize.value() < 0) {
            return tl::make_unexpected(Error{{}, "[@max_inner_dims] arguments must be positive!"});
        }
        ret.size[i] = dimSize.value();
    }

    return ret;
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<MaxInnerDims>(MAX_INNER_DIMS,
                                                                    parseMaxInnerDims);
}
}  // namespace
