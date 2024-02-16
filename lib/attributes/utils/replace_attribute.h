#pragma once

#include <string>

namespace clang {
class Decl;
class Stmt;
class Attr;
}  // namespace clang

namespace oklt {

class SessionStage;

bool handleGlobalConstant(const clang::Decl* decl, SessionStage& s, const std::string& qualifier);
bool handleGlobalFunction(const clang::Decl* decl,
                          SessionStage& s,
                          const std::string& funcQualifier);

bool handleTileAttribute(const clang::Attr* a, const clang::Stmt* d, SessionStage& s);

}  // namespace oklt
