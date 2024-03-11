#include "core/utils/type_converter.h"
#include <oklt/core/error.h>

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

inline DatatypeCategory toOklDatatypeCategory(const clang::QualType& qt) {
    auto qt_ = [](const clang::QualType qt) {
        if (qt->isPointerType()) {
            return qt->getPointeeType();
        }
        return qt;
    }(qt);

    if (qt_->isBuiltinType()) {
        return DatatypeCategory::BUILTIN;
    }
    if (qt_->isStructureType()) {
        return DatatypeCategory::STRUCT;
    }
    if (qt_->isConstantArrayType()) {
        return DatatypeCategory::TUPLE;
    }
    return DatatypeCategory::CUSTOM;
}

tl::expected<StructFieldInfo, std::error_code> toOklStructFieldInfo(const clang::FieldDecl& var) {
    if (var.isTemplated()) {
        return tl::make_unexpected(std::error_code());
    }

    auto qt = var.getType();
    bool is_const = qt.isConstQualified();
    if (isPointer(var)) {
        is_const = isConstPointer(var) || isPointerToConst(var);
    }
    StructFieldInfo res{.dtype = toOklDataType(var).value(), .name = var.getNameAsString()};
    return res;
}

tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const VarDecl& var) {
    // inline tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const clang::VarDecl& var) {
    // templated arg is abstract
    if (var.isTemplated()) {
        return tl::make_unexpected(std::error_code());
    }

    auto qt = var.getType();
    bool is_const = qt.isConstQualified();
    if (isPointer(var)) {
        is_const = isConstPointer(var) || isPointerToConst(var);
    }
    ArgumentInfo res{.is_const = is_const,
                     .dtype = toOklDataType(var).value(),
                     .name = var.getNameAsString(),
                     .is_ptr = qt->isPointerType()};
    return res;
}

}  // namespace oklt
