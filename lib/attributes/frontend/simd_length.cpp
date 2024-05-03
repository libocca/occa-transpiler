#include "attributes/attribute_names.h"
#include "attributes/frontend/params/loop.h"
#include "attributes/utils/parser.h"
#include "attributes/utils/parser_impl.hpp"

#include "core/handler_manager/parse_handler.h"

#include <clang/Basic/DiagnosticSema.h>
#include <clang/Sema/ParsedAttr.h>
#include <clang/Sema/Sema.h>

namespace {
using namespace clang;
using namespace oklt;

constexpr ParsedAttrInfo::Spelling SIMD_LENGTH_ATTRIBUTE_SPELLINGS[] = {
    {ParsedAttr::AS_CXX11, SIMD_LENGTH_NAME},
    {ParsedAttr::AS_GNU, SIMD_LENGTH_NAME}};

struct SimdLengthAttribute : public ParsedAttrInfo {
    SimdLengthAttribute() {
        NumArgs = 1;
        OptArgs = 0;
        Spellings = SIMD_LENGTH_ATTRIBUTE_SPELLINGS;
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

HandleResult parseSimdLengthAttrParams(SessionStage& stage,
                                       const clang::Attr& attr,
                                       OKLParsedAttr& data) {
    if (!data.kwargs.empty()) {
        return tl::make_unexpected(Error{{}, "[@simd_length] does not take kwargs"});
    }

    if (data.args.size() != 1) {
        return tl::make_unexpected(Error{{}, "[@simd_length] takes one argument"});
    }

    auto sz = data.get<int>(0);
    if (!sz) {
        return tl::make_unexpected(Error{{}, "[@simd_length] take an integer argument"});
    } else if (sz.value() < 0) {
        return tl::make_unexpected(Error{{}, "[@simd_length] arguments must be positive!"});
    }

    auto ret = AttributedLoopSimdLength{.size = sz.value_or(-1)};

    return ret;
}

__attribute__((constructor)) void registerSimdLengthAttrFrontend() {
    registerAttrFrontend<SimdLengthAttribute>(SIMD_LENGTH_NAME, parseSimdLengthAttrParams);
}
}  // namespace
