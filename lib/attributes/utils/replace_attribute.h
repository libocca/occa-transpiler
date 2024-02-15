#pragma once

#include <string>

namespace clang {
class Decl;
}

namespace oklt {

class SessionStage;

bool handleGlobalConstant(const clang::Decl* decl, SessionStage& s, const std::string &qualifier);
bool handleGlobalFunction(const clang::Decl* decl, SessionStage& s, const std::string &funcQualifier);

}
