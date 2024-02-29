#include "attributes/frontend/params/barrier.h"
#include "attributes/frontend/params/tile.h"
#include "core/attribute_manager/result.h"

namespace clang {
class Attr;
class Stmt;
class ForStmt;
class Decl;
class FunctionDecl;
class ParmVarDecl;
}  // namespace clang

namespace oklt {
class SessionStage;
}

namespace oklt::cuda_subset {
HandleResult handleTileAttribute(const clang::Attr&,
                                 const clang::ForStmt&,
                                 const TileParams* params,
                                 SessionStage&);
HandleResult handleInnerAttribute(const clang::Attr&,
                                  const clang::ForStmt&,
                                  const AttributedLoop* params,
                                  SessionStage&);
HandleResult handleOuterAttribute(const clang::Attr&,
                                  const clang::ForStmt&,
                                  const AttributedLoop* params,
                                  SessionStage&);
HandleResult handleAtomicAttribute(const clang::Attr&, const clang::Stmt&, SessionStage&);

HandleResult handleKernelAttribute(const clang::Attr&, const clang::FunctionDecl&, SessionStage&);
HandleResult handleSharedAttribute(const clang::Attr&, const clang::Decl&, SessionStage&);
HandleResult handleRestrictAttribute(const clang::Attr&, const clang::ParmVarDecl&, SessionStage&);

HandleResult handleExclusiveAttribute(const clang::Attr&, const clang::Decl&, SessionStage&);
HandleResult handleBarrierAttribute(const clang::Attr&,
                                    const clang::Stmt&,
                                    const oklt::AttributedBarrier*,
                                    SessionStage&);

}  // namespace oklt::cuda_subset
