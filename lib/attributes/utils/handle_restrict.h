#pragma once

#include <string>

namespace clang {
class Attr;
class Decl;
}  // namespace clang

namespace oklt {
class SessionStage;

bool handleRestrictAttribute(const clang::Attr* a,
                             const clang::Decl* d,
                             SessionStage& s,
                             const std::string& replaceQualifier);

}  // namespace oklt
