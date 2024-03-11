#include <clang/AST/AST.h>
#include <oklt/core/kernel_metadata.h>
#include "core/utils/var_decl.h"

#include <tl/expected.hpp>

namespace clang {
struct QualType;
struct VarDecl;
struct ForStmt;
struct ASTContext;
}  // namespace clang

namespace oklt {
struct Error;
struct LoopMetaData;

tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const clang::VarDecl&);
tl::expected<StructFieldInfo, std::error_code> toOklStructFieldInfo(const clang::FieldDecl&);
DatatypeCategory toOklDatatypeCategory(const clang::QualType&);

template <typename DeclType>
tl::expected<DataType, std::error_code> toOklDataType(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    // templated arg is abstract
    if (var.isTemplated()) {
        return tl::make_unexpected(std::error_code());
    }

    // Find correct unqualified type
    auto qt = var.getType();
    auto unqualifiedType = qt.getUnqualifiedType();

    // Base type = pointed type for pointer or element type for array
    auto baseType = unqualifiedType;
    if (baseType->isPointerType()) {
        baseType = baseType->getPointeeType().getUnqualifiedType();
    }
    if (isConstantSizeArray(var)) {
        baseType = clang::dyn_cast_or_null<clang::ArrayType>(baseType)
                       ->getElementType()
                       .getUnqualifiedType();
    }
    std::string name = baseType.getCanonicalType().getAsString();

    auto type = toOklDatatypeCategory(qt);

    DataType res{.name = name, .type = type};
    if (type == DatatypeCategory::CUSTOM) {
        res.bytes = static_cast<int>(var.getASTContext().getTypeSize(qt));
    }
    if (type == DatatypeCategory::STRUCT) {
        auto* typePtr = var.getType().getTypePtr();
        const auto* structDecl = typePtr->getAsCXXRecordDecl();
        if (typePtr->isPointerType()) {
            structDecl = typePtr->getPointeeCXXRecordDecl();
        }
        if (!structDecl) {
            return tl::make_unexpected(std::error_code());
        }

        for (const auto* field : structDecl->fields()) {
            auto n = field->getNameAsString();
            auto fieldDataType = toOklStructFieldInfo(*field);
            if (!fieldDataType) {
                return tl::make_unexpected(fieldDataType.error());
            }
            res.fields.push_back(fieldDataType.value());
        }
    }
    if (type == DatatypeCategory::TUPLE) {
        res.tupleElementType = toOklDatatypeCategory(baseType);
        auto arraySize =
            clang::dyn_cast_or_null<clang::ConstantArrayType>(unqualifiedType)->getSize();
        if (arraySize.isIntN(sizeof(int64_t) * 8)) {  // Check if APInt fits within the range of int
            res.tupleSize = arraySize.getSExtValue();  // Convert APInt to int
        } else {
            // APInt value too large to fit into int64_t
            return tl::make_unexpected(std::error_code());
        }
    }
    return res;
}
}  // namespace oklt
