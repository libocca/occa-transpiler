#include "attributes/utils/replace_attribute.h"
#include "core/transpiler_session/session_stage.h"

#include <clang/AST/AST.h>

namespace oklt {
using namespace clang;

bool isConstantSizeArray(const VarDecl& var) {
    return var.getType().getTypePtr()->isConstantArrayType();
}

bool isPointer(const VarDecl& var) {
    return var.getType()->isPointerType();
}

bool isPointerToConst(const VarDecl& var) {
    return isPointer(var) && var.getType()->getPointeeType().isLocalConstQualified();
}

bool isConstPointer(const VarDecl& var) {
    return isPointer(var) && var.getType().isLocalConstQualified();
}

bool isConstPointerToConst(const VarDecl& var) {
    return isPointerToConst(var) && isConstPointer(var);
}

bool isGlobalConstVariable(const VarDecl& var) {
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
        llvm::outs() << var.getDeclName().getAsString() << " is not constant\n";
        return false;
    }

    return true;
}

std::string getNewDeclStrConstantArray(const VarDecl& var, const std::string& qualifier) {
    auto* arrDecl = dyn_cast<ConstantArrayType>(var.getType().getTypePtr());
    auto unqualifiedTypeStr = arrDecl->getElementType().getLocalUnqualifiedType().getAsString();

    auto type = arrDecl->getElementType();
    type.removeLocalConst();
    auto qualifiers = type.getQualifiers();

    auto varName = var.getDeclName().getAsString();  // Name of variable

    std::string newDeclStr;
    if (qualifiers.hasQualifiers()) {
        auto noConstQualifiersStr = qualifiers.getAsString();
        newDeclStr =
            noConstQualifiersStr + " " + qualifier + " " + unqualifiedTypeStr + " " + varName;
    } else {
        newDeclStr = qualifier + " " + unqualifiedTypeStr + " " + varName;
    }
    return newDeclStr;
}

std::string getNewDeclStrVariable(const VarDecl& var, const std::string& qualifier) {
    auto unqualifiedTypeStr = var.getType().getLocalUnqualifiedType().getAsString();

    auto type = var.getType();
    type.removeLocalConst();
    auto qualifiers = type.getQualifiers();

    auto VarName = var.getDeclName().getAsString();  // Name of variable

    std::string newDeclStr;
    if (qualifiers.hasQualifiers()) {
        auto noConstQualifiersStr = qualifiers.getAsString();
        newDeclStr =
            noConstQualifiersStr + " " + qualifier + " " + unqualifiedTypeStr + " " + VarName;
    } else {
        newDeclStr = qualifier + " " + unqualifiedTypeStr + " " + VarName;
    }
    return newDeclStr;
}

std::string getNewDeclStrPointerToConst(const VarDecl& var, const std::string& qualifier) {
    auto type = var.getType();

    auto unqualifiedPointeeType = type->getPointeeType();
    unqualifiedPointeeType.removeLocalConst();
    auto unqualifiedPointeeTypeStr = unqualifiedPointeeType.getAsString();

    auto varName = var.getDeclName().getAsString();

    std::string newDeclStr;
    if (type.hasQualifiers()) {
        auto qualifiersStr = type.getQualifiers().getAsString();
        newDeclStr =
            qualifier + " " + unqualifiedPointeeTypeStr + " * " + qualifiersStr + " " + varName;
    } else {
        newDeclStr = qualifier + " " + unqualifiedPointeeTypeStr + " * " + varName;
    }
    return newDeclStr;
}
}  // namespace oklt
