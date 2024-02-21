#pragma once

namespace clang {
class Attr;
class Stmt;
}  // namespace clang

namespace oklt {

class SessionStage;
bool handleAtomicAttribute(const clang::Attr* attr, const clang::Stmt* stmt, SessionStage& stage);
}  // namespace oklt
