#include "core/utils/type_converter.h"
#include <oklt/core/error.h>

#include <clang/AST/AST.h>
#include <memory>
#include "oklt/core/kernel_metadata.h"

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

clang::QualType getBaseType(const clang::QualType& type) {
    auto baseType = type.getUnqualifiedType();
    if (baseType->isPointerType()) {
        baseType = baseType->getPointeeType().getUnqualifiedType();
    }
    if (type.getTypePtr()->isConstantArrayType()) {
        baseType = clang::dyn_cast_or_null<clang::ConstantArrayType>(baseType)
                       ->getElementType()
                       .getUnqualifiedType();
    }
    return baseType;
}

tl::expected<void, std::error_code> fillStructFields(std::list<StructFieldInfo>& fields,
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
        fields.push_back(fieldDataType.value());
    }
    return {};
}

tl::expected<void, std::error_code> fillTupleElement(
    const clang::QualType& type,
    const std::shared_ptr<TupleElementDataType>& tupleElementDType) {
    auto elementType = getBaseType(type);
    tupleElementDType->typeCategory = toOklDatatypeCategory(elementType);
    tupleElementDType->name = elementType.getCanonicalType().getAsString();

    // In case element type is struct, we must fill it's fields
    if (tupleElementDType->typeCategory == DatatypeCategory::STRUCT) {
        auto fillRes =
            detail::fillStructFields(tupleElementDType->fields, elementType.getTypePtr());
        if (!fillRes) {
            return fillRes;
        }
    }
    if (tupleElementDType->typeCategory == DatatypeCategory::TUPLE) {
        tupleElementDType->tupleElementDType = std::make_shared<TupleElementDataType>();
        auto fillStatus = fillTupleElement(elementType, tupleElementDType->tupleElementDType);
        if (!fillStatus) {
            return fillStatus;
        }
    }
    auto arraySize = clang::dyn_cast_or_null<clang::ConstantArrayType>(type)->getSize();
    if (arraySize.isIntN(sizeof(int64_t) * 8)) {  // Check if APInt fits within the range of int
        tupleElementDType->tupleSize = arraySize.getSExtValue();  // Convert APInt to int
    } else {
        // APInt value too large to fit into int64_t
        return tl::make_unexpected(std::error_code());
    }
    return {};
}
}  // namespace detail

}  // namespace oklt
