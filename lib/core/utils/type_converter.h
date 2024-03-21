#pragma once

#include <clang/AST/AST.h>
#include <oklt/core/kernel_metadata.h>
#include "clang/AST/Type.h"
#include "core/utils/var_decl.h"

#include <memory>
#include <tl/expected.hpp>

namespace clang {
struct QualType;
struct FunctionDecl;
struct VarDecl;
struct ForStmt;
struct ASTContext;
}  // namespace clang

namespace oklt {
struct Error;

tl::expected<KernelInfo, std::error_code> toOklKernelInfo(const clang::FunctionDecl&,
                                                          const std::string& suffix = "");
tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const clang::VarDecl&);
tl::expected<StructFieldInfo, std::error_code> toOklStructFieldInfo(const clang::FieldDecl&);
DatatypeCategory toOklDatatypeCategory(const clang::QualType&);

// Functions in this namespace are not ment to be called outside of this and .cpp file. But since
// They are used in header, it's impossible to hide then in anon namespace in .cpp file :( )
namespace detail {
//  Base type = pointed type for pointer or element type for array, otherwise just type of 'var'
clang::QualType getBaseType(const clang::QualType& type);
tl::expected<void, std::error_code> fillStructFields(std::list<StructFieldInfo>& fields,
                                                     const clang::Type* structType);
tl::expected<void, std::error_code> fillTupleElement(
    const clang::QualType& qt,
    const std::shared_ptr<TupleElementDataType>& tupleElementType);
}  // namespace detail

template <typename DeclType>
tl::expected<DataType, std::error_code> toOklDataType(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    // templated arg is abstract
    if (var.isTemplated()) {
        return tl::make_unexpected(std::error_code());
    }

    // Find correct unqualified type
    auto type = var.getType();
    auto baseType = detail::getBaseType(type);

    std::string name = baseType.getCanonicalType().getAsString();
    auto typeCategory = toOklDatatypeCategory(type);

    DataType res{.name = name, .typeCategory = typeCategory};
    if (typeCategory == DatatypeCategory::CUSTOM) {
        res.bytes = static_cast<int>(var.getASTContext().getTypeSize(type));
    }
    if (typeCategory == DatatypeCategory::STRUCT) {
        // Fill type of each struct field
        auto fillRes = detail::fillStructFields(res.fields, type.getTypePtr());
        if (!fillRes) {
            return tl::make_unexpected(fillRes.error());
        }
    }
    if (typeCategory == DatatypeCategory::TUPLE) {
        res.tupleElementDType = std::make_shared<TupleElementDataType>();
        auto fillRes = detail::fillTupleElement(type, res.tupleElementDType);
        if (!fillRes) {
            return tl::make_unexpected(fillRes.error());
        }
    }
    return res;
}
}  // namespace oklt
