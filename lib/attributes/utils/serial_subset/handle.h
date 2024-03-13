#include "attributes/frontend/params/loop.h"
#include "attributes/frontend/params/tile.h"
#include "core/attribute_manager/result.h"

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
HandleResult handleTileAttribute(const clang::Attr&,
                                 const clang::ForStmt&,
                                 const TileParams*,
                                 SessionStage&);
HandleResult handleInnerAttribute(const clang::Attr&,
                                  const clang::ForStmt&,
                                  const AttributedLoop* params,
                                  SessionStage&);
HandleResult handleOuterAttribute(const clang::Attr&,
                                  const clang::ForStmt&,
                                  const AttributedLoop* params,
                                  SessionStage&);

HandleResult handleKernelAttribute(const clang::Attr&, const clang::FunctionDecl&, SessionStage&);
HandleResult handleSharedAttribute(const clang::Attr&, const clang::Decl&, SessionStage&);
HandleResult handleRestrictAttribute(const clang::Attr&, const clang::ParmVarDecl&, SessionStage&);

HandleResult handleExclusiveDeclAttribute(const clang::Attr&, const clang::VarDecl&, SessionStage&);
HandleResult handleExclusiveExprAttribute(const clang::Attr&,
                                          const clang::DeclRefExpr&,
                                          SessionStage&);

HandleResult handleEmptyDeclAttribute(const clang::Attr&,
                                      const clang::Decl&,
                                      const std::any*,
                                      SessionStage&);
HandleResult handleEmptyStmtAttribute(const clang::Attr&,
                                      const clang::Stmt&,
                                      const std::any*,
                                      SessionStage&);

}  // namespace oklt::serial_subset
