#include "core/utils/type_converter.h"
#include <oklt/core/error.h>

#include <clang/AST/AST.h>

namespace oklt {
tl::expected<DataType, std::error_code> toOklDataType(const clang::VarDecl& var) {
    // templated arg is abstract
    if (var.isTemplated()) {
        return tl::make_unexpected(std::error_code());
    }

    auto qt = var.getType();
    return DataType{.name = qt.getAsString(),
                    .type = toOklDatatypeCategory(qt),
                    .bytes = static_cast<int>(var.getASTContext().getTypeSize(qt))};
}

tl::expected<ArgumentInfo, std::error_code> toOklArgInfo(const clang::VarDecl& var) {
    // templated arg is abstract
    if (var.isTemplated()) {
        return tl::make_unexpected(std::error_code());
    }

    auto qt = var.getType();
    return ArgumentInfo{.is_const = qt.isConstQualified(),
                        .dtype = toOklDataType(var).value(),
                        .name = var.getNameAsString(),
                        .is_ptr = qt->isPointerType()};
}

DatatypeCategory toOklDatatypeCategory(const clang::QualType& qt) {
    auto qt_ = [](const clang::QualType qt) {
        if (qt->isPointerType()) {
            return qt->getPointeeType();
        }
        return qt;
    }(qt);

    if (qt_->isBuiltinType()) {
        return DatatypeCategory::BUILTIN;
    }
    return DatatypeCategory::CUSTOM;
}

}  // namespace oklt
