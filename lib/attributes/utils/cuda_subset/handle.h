#include "attributes/frontend/params/barrier.h"
#include "attributes/frontend/params/tile.h"
#include "core/handler_manager/result.h"

namespace clang {
class Attr;
class Stmt;
class ForStmt;
class Decl;
class VarDecl;
class TypedefDecl;
class FunctionDecl;
class ParmVarDecl;
}  // namespace clang

namespace oklt {
class SessionStage;
}

namespace oklt::cuda_subset {
HandleResult handleTileAttribute(SessionStage&,
                                 const clang::ForStmt&,
                                 const clang::Attr&,
                                 const TileParams* params);
HandleResult handleInnerAttribute(SessionStage&,
                                  const clang::ForStmt&,
                                  const clang::Attr&,
                                  const AttributedLoop* params);
HandleResult handleOuterAttribute(SessionStage&,
                                  const clang::ForStmt&,
                                  const clang::Attr&,
                                  const AttributedLoop* params);
HandleResult handleAtomicAttribute(SessionStage&, const clang::Stmt&, const clang::Attr&);

HandleResult handleKernelAttribute(SessionStage&, const clang::FunctionDecl&, const clang::Attr&);
HandleResult handleSharedDeclAttribute(SessionStage&, const clang::Decl&, const clang::Attr&);
HandleResult handleSharedTypeAttribute(SessionStage&,
                                       const clang::TypedefDecl&,
                                       const clang::Attr&);
HandleResult handleSharedVarAttribute(SessionStage&, const clang::VarDecl&, const clang::Attr&);
HandleResult handleRestrictAttribute(SessionStage&, const clang::Decl&, const clang::Attr&);

HandleResult handleExclusiveAttribute(SessionStage&, const clang::Decl&, const clang::Attr&);
oklt::HandleResult handleBarrierAttribute(SessionStage&,
                                          const clang::Stmt&,
                                          const clang::Attr&,
                                          const oklt::AttributedBarrier*);

}  // namespace oklt::cuda_subset
