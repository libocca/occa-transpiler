#include "attributes/attribute_names.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"
#include "core/attribute_manager/parsed_attribute_info_base.h"
#include "params/empty_params.h"

namespace {

using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling NOBARRIER_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, "nobarrier"},
    {ParsedAttr::AS_CXX11, NOBARRIER_ATTR_NAME},
    {ParsedAttr::AS_GNU, "okl_nobarrier"}};

struct NoBarrierAttribute : public ParsedAttrInfoBase {
    NoBarrierAttribute() {
        Spellings = NOBARRIER_ATTRIBUTE_SPELLINGS;
        NumArgs = 1;
        OptArgs = 0;
        IsStmt = true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Stmt& stmt) const override {
        if (!isa<NullStmt>(stmt)) {
            sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
                << attr << attr.isDeclspecAttribute() << "empty statement";
            return false;
        }
        return true;
    }

    bool diagAppertainsTo(clang::Sema& sema,
                          const clang::ParsedAttr& attr,
                          const clang::Decl& decl) const override {
        // INFO: fail for all decls
        sema.Diag(attr.getLoc(), diag::err_attribute_wrong_decl_type_str)
            << attr << attr.isDeclspecAttribute() << "empty statements";
        return false;
    }
};

ParseResult parseNoBarrierAttrParams(const clang::Attr& attr,
                                     OKLParsedAttr& data,
                                     SessionStage& stage) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@nobarrier] does not take kwargs"});
    }
    if (!data.args.empty()) {
        return tl::make_unexpected(Error{{}, "[@nobarrier] does not take arguments"});
    }

    return EmptyParams{};
}

__attribute__((constructor)) void registerAttrFrontend() {
    AttributeManager::instance().registerAttrFrontend<NoBarrierAttribute>(NOBARRIER_ATTR_NAME,
                                                                          parseNoBarrierAttrParams);
}
}  // namespace
