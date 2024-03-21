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
    bool is_ptr = false;
    if (isPointer(var)) {
        is_ptr = true;
        is_const = isConstPointer(var) || isPointerToConst(var);
    }
    ArgumentInfo res{.is_const = is_const,
                     .dtype = toOklDataType(var).value(),
                     .name = var.getNameAsString(),
                     .is_ptr = is_ptr};
    return res;
}

tl::expected<KernelInfo, std::error_code> toOklKernelInfo(const FunctionDecl& fd,
                                                          const std::string& suffix) {
    KernelInfo ret;
    ret.name = fd.getNameAsString() + suffix;

    for (auto param : fd.parameters()) {
        if (!param) {
            return tl::make_unexpected(std::error_code());
        }
        auto arg = toOklArgInfo(*param);
        if (!arg) {
            return tl::make_unexpected(arg.error());
        }
        ret.args.emplace_back(std::move(arg.value()));
    }

    return ret;
}

namespace detail {
tl::expected<void, std::error_code> fillStructFields(DataType& dt,
                                                     const clang::Type* structTypePtr) {
    const auto* structDecl = structTypePtr->getAsCXXRecordDecl();
    if (structTypePtr->isPointerType()) {
        structDecl = structTypePtr->getPointeeCXXRecordDecl();
    }
    if (!structDecl) {
        return tl::make_unexpected(std::error_code());
    }

    for (const auto* field : structDecl->fields()) {
        auto fieldDataType = toOklStructFieldInfo(*field);
        if (!fieldDataType) {
            return tl::make_unexpected(fieldDataType.error());
        }
        dt.fields.push_back(fieldDataType.value());
    }
    return {};
}
}  // namespace detail

}  // namespace oklt
