#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"
#include "core/handler_manager/result.h"

namespace clang {
class Attr;
class Stmt;
class ForStmt;
class DeclRefExpr;
class Decl;
class VarDecl;
class FunctionDecl;
class ParmVarDecl;
}  // namespace clang

namespace oklt {
class SessionStage;
}

namespace oklt::serial_subset {
HandleResult handleTileAttribute(SessionStage&,
                                 const clang::ForStmt&,
                                 const clang::Attr&,
                                 const TileParams*);
HandleResult handleInnerAttribute(SessionStage&,
                                  const clang::ForStmt&,
                                  const clang::Attr&,
                                  const AttributedLoop* params);
HandleResult handleOuterAttribute(SessionStage&,
                                  const clang::ForStmt&,
                                  const clang::Attr&,
                                  const AttributedLoop* params);

HandleResult handleKernelAttribute(SessionStage&, const clang::FunctionDecl&, const clang::Attr&);
HandleResult handleSharedAttribute(SessionStage&, const clang::Decl&, const clang::Attr&);
HandleResult handleRestrictAttribute(SessionStage&, const clang::Decl&, const clang::Attr&);

HandleResult handleExclusiveDeclAttribute(SessionStage&, const clang::VarDecl&, const clang::Attr&);
HandleResult handleExclusiveExprAttribute(SessionStage&,
                                          const clang::DeclRefExpr&,
                                          const clang::Attr&);

HandleResult handleEmptyDeclAttribute(SessionStage&,
                                      const clang::Decl&,
                                      const clang::Attr&,
                                      const std::any*);
HandleResult handleEmptyStmtAttribute(SessionStage&,
                                      const clang::Stmt&,
                                      const clang::Attr&,
                                      const std::any*);

}  // namespace oklt::serial_subset
