#pragma once

#include <clang/AST/Decl.h>

#include <string>

namespace clang {
class VarDecl;
}

namespace oklt {
template <typename DeclType>
bool isConstantSizeArray(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    return var.getType().getTypePtr()->isConstantArrayType();
}

template <typename DeclType>
bool isArray(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    return var.getType().getTypePtr()->isArrayType();
}

template <typename DeclType>
bool isPointer(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    return var.getType()->isPointerType();
}

template <typename DeclType>
bool isPointerToConst(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    return isPointer(var) && var.getType()->getPointeeType().isLocalConstQualified();
}

template <typename DeclType>
bool isConstPointer(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    return isPointer(var) && var.getType().isLocalConstQualified();
}

template <typename DeclType>
bool isConstPointerToConst(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    return isPointerToConst(var) && isConstPointer(var);
}

template <typename DeclType>
bool isGlobalConstVariable(const DeclType& var) {
    static_assert(std::is_base_of_v<clang::Decl, DeclType>);
    // Skip constexpr
    if (var.isConstexpr()) {
        return false;
    }

    // Should be global variable
    if (var.isLocalVarDecl() && !var.hasGlobalStorage()) {
        return false;
    }

    // pointer to const
    if (isPointer(var)) {
        return isPointerToConst(var);
    }

    auto type = var.getType();
    // Should be constant qualified
    if (!(type.isLocalConstQualified() || type.isConstant(var.getASTContext()))) {
        return false;
    }

    return true;
}
std::string getNewDeclStrArray(const clang::VarDecl& var, const std::string& qualifier);
std::string getNewDeclStrVariable(const clang::VarDecl& var, const std::string& qualifier);
std::string getNewDeclStrPointerToConst(const clang::VarDecl& var, const std::string& qualifier);
}  // namespace oklt
