#pragma once

#include <string>

namespace clang {
class Attr;
class SourceRange;
}  // namespace clang

namespace oklt {

class SessionStage;

clang::SourceRange getAttrFullSourceRange(const clang::Attr& attr);
bool removeAttribute(const clang::Attr& attr, SessionStage& stage);
bool isOklAttribute(const clang::Attr& attr);
std::string getOklAttrFullName(const clang::Attr& attr);
}  // namespace oklt
