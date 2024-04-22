#pragma once

namespace clang {
class Attr;
class SourceRange;
}  // namespace clang

namespace oklt {

class SessionStage;

clang::SourceRange getAttrFullSourceRange(const clang::Attr& attr);
bool removeAttribute(SessionStage& stage, const clang::Attr& attr);
bool isOklAttribute(const clang::Attr& attr);
}  // namespace oklt
