#pragma once

#include "core/handler_manager/result.h"

#include <clang/AST/AST.h>
#include <clang/AST/Attr.h>

namespace clang {
class Stmt;
}  // namespace clang

namespace oklt {
class SessionStage;

HandleResult emptyHandleStmtAttribute(SessionStage&, const clang::Stmt&, const clang::Attr&);

HandleResult emptyHandleDeclAttribute(SessionStage&, const clang::Decl&, const clang::Attr&);

HandleResult defaultHandleSharedStmtAttribute(SessionStage&,
                                              const clang::Stmt&,
                                              const clang::Attr&);

HandleResult defaultHandleExclusiveStmtAttribute(SessionStage&,
                                                 const clang::Stmt&,
                                                 const clang::Attr&);

HandleResult defaultHandleSharedDeclAttribute(SessionStage&,
                                              const clang::Decl&,
                                              const clang::Attr&);

HandleResult defaultHandleExclusiveDeclAttribute(SessionStage&,
                                                 const clang::Decl&,
                                                 const clang::Attr&);

}  // namespace oklt
