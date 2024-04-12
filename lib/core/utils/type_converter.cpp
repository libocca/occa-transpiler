#include "core/utils/type_converter.h"
#include <oklt/core/error.h>
#include <oklt/core/kernel_metadata.h>

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

namespace {

std::string getTypeName(const QualType& type) {
    auto strippedType = type;
    if (strippedType->isPointerType()) {
        strippedType = strippedType->getPointeeType().getUnqualifiedType();
    }
    auto res = strippedType.getCanonicalType().getAsString();
    if (res == "_Bool") {
        return "bool";
    }
    return res;
}

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

tl::expected<void, std::error_code> fillEnumNames(std::vector<std::string>& enumNames,
                                                  const Type* type_) {
    const Type* type = type_;
    if (type->isPointerType()) {
        type = type->getPointeeType().getTypePtr();
    }

    auto enumType = dyn_cast<EnumType>(type->getCanonicalTypeUnqualified());
    if (!enumType) {
        return tl::make_unexpected(std::error_code());
    }
    const EnumDecl* enumDecl = enumType->getDecl();
    if (!enumDecl) {
        return tl::make_unexpected(std::error_code());
    }

    for (const auto& enumerator : enumDecl->enumerators()) {
        enumNames.push_back(enumerator->getNameAsString());
    }

    return {};
}

tl::expected<DataType, std::error_code> toOklDataTypeImpl(const QualType& type, ASTContext& ctx);
tl::expected<void, std::error_code> fillTupleElement(
    const clang::QualType& type,
    const std::shared_ptr<TupleElementDataType>& tupleElementDType,
    ASTContext& ctx) {
    auto elementType = getBaseType(type);
    auto elementDType = toOklDataTypeImpl(elementType, ctx);
    if (!elementDType) {
        return tl::make_unexpected(elementDType.error());
    }
    tupleElementDType->elementDType = elementDType.value();

    auto arraySize = clang::dyn_cast_or_null<clang::ConstantArrayType>(type)->getSize();
    if (arraySize.isIntN(sizeof(int64_t) * 8)) {  // Check if APInt fits within the range of int
        tupleElementDType->tupleSize = arraySize.getSExtValue();  // Convert APInt to int
    } else {
        // APInt value too large to fit into int64_t
        return tl::make_unexpected(std::error_code());
    }
    return {};
}

tl::expected<DataType, std::error_code> toOklDataTypeImpl(const QualType& type, ASTContext& ctx) {
    auto baseType = getBaseType(type);
    auto name = getTypeName(baseType);
    auto anotherName =
        getTypeName(QualType(baseType.getTypePtr()->getUnqualifiedDesugaredType(), 0));
    auto typeCategory = toOklDatatypeCategory(type);

    DataType res{.name = name, .typeCategory = typeCategory};
    switch (typeCategory) {
        case DatatypeCategory::CUSTOM: {
            res.bytes = static_cast<int>(ctx.getTypeSize(type));
            break;
        }
        case DatatypeCategory::STRUCT: {
            // Fill type of each struct field
            auto fillRes = fillStructFields(res.fields, type.getTypePtr());
            if (!fillRes) {
                return tl::make_unexpected(fillRes.error());
            }
            break;
        }
        case DatatypeCategory::TUPLE: {
            res.tupleElementDType = std::make_shared<TupleElementDataType>();
            auto fillRes = fillTupleElement(type, res.tupleElementDType, ctx);
            if (!fillRes) {
                return tl::make_unexpected(fillRes.error());
            }
            break;
        }
        case DatatypeCategory::ENUM: {
            auto fillRes = fillEnumNames(res.enumNames, type.getTypePtr());
            if (!fillRes) {
                return tl::make_unexpected(fillRes.error());
            }
            break;
        }
        default: {
        }
    }
    return res;
}

template <typename DeclType>
tl::expected<DataType, std::error_code> toOklDataTypeImpl(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    // templated arg is abstract
    if (var.isTemplated()) {
        return tl::make_unexpected(std::error_code());
    }

    // Find correct unqualified type
    auto type = var.getType();
    if (isa<ParmVarDecl>(var)) {
        // This gets type of array before decay
        type = dyn_cast<ParmVarDecl>(&var)->getOriginalType();
    }
    return toOklDataTypeImpl(type, var.getASTContext());
}

}  // namespace

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
    if (qt_->isEnumeralType()) {
        return DatatypeCategory::ENUM;
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
    auto dt = toOklDataType(var);
    if (!dt) {
        return tl::make_unexpected(dt.error());
    }

    StructFieldInfo res{.dtype = dt.value(), .name = var.getNameAsString()};
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

tl::expected<DataType, std::error_code> toOklDataType(const clang::VarDecl& var) {
    return toOklDataTypeImpl(var);
}

tl::expected<DataType, std::error_code> toOklDataType(const clang::FieldDecl& var) {
    return toOklDataTypeImpl(var);
}

}  // namespace oklt
