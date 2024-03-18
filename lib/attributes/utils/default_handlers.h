#pragma  once

#include "core/attribute_manager/result.h"

#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>

namespace clang {
class Stmt;
}  // namespace clang

namespace oklt {
class SessionStage;

HandleResult emptyHandleStmtAttribute(const clang::Attr&, const clang::Stmt&, SessionStage&);

HandleResult emptyHandleDeclAttribute(const clang::Attr&, const clang::Decl&, SessionStage&);

HandleResult defaultHandleSharedStmtAttribute(const clang::Attr&,
                                              const clang::Stmt&,
                                              SessionStage&);

}  // namespace oklt
