#pragma once

#include <string>

namespace clang {
class VarDecl;
}

namespace oklt {
bool isConstantSizeArray(const clang::VarDecl* var);
bool isPointer(const clang::VarDecl* var);
bool isPointerToConst(const clang::VarDecl* var);
bool isConstPointer(const clang::VarDecl* var);
bool isConstPointerToConst(const clang::VarDecl* var);
bool isGlobalConstVariable(const clang::VarDecl* var);
std::string getNewDeclStrConstantArray(const clang::VarDecl* var, const std::string &qualifier);
std::string getNewDeclStrVariable(const clang::VarDecl* var, const std::string &qualifier);
std::string getNewDeclStrPointerToConst(const clang::VarDecl* var, const std::string &qualifier);
}
