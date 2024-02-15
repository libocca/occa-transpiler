#pragma once

namespace clang {
class Attr;
class SourceRange;
}

namespace oklt {

class SessionStage;

clang::SourceRange getAttrFullSourceRange(const clang::Attr& attr);
bool removeAttribute(const clang::Attr *attr, SessionStage &stage);
}
