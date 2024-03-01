#pragma once

#include "core/attribute_manager/result.h"

#include <clang/AST/AST.h>
namespace oklt {

struct OklSemaCtx;
class SessionStage;

HandleResult runDefaultPreActionDecl(const clang::Attr* attr,
                                     const clang::Decl& decl,
                                     OklSemaCtx& sema,
                                     SessionStage& stage);

HandleResult runDefaultPreActionStmt(const clang::Attr* attr,
                                     const clang::Stmt& stmt,
                                     OklSemaCtx& sema,
                                     SessionStage& stage);

HandleResult runDefaultPostActionDecl(const clang::Attr* attr,
                                      const clang::Decl& decl,
                                      OklSemaCtx& sema,
                                      SessionStage& stage);

HandleResult runDefaultPostActionStmt(const clang::Attr* attr,
                                      const clang::Stmt& stmt,
                                      OklSemaCtx& sema,
                                      SessionStage& stage);

}  // namespace oklt
