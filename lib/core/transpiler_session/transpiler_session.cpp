#include <clang/Basic/SourceManager.h>

#include "oklt/core/transpiler_session/transpiler_session.h"
#include "oklt/core/utils/format.h"


namespace oklt {
using namespace clang;

TranspilerSession::TranspilerSession(TRANSPILER_TYPE backend)
  :targetBackend(backend)
  ,transpiledCode()
{}


SessionStage::SessionStage(TranspilerSession &globalSession,
                                     ASTContext &ctx)
    :_globalSession(globalSession)
    , _ctx(ctx)
    ,_rewriter(ctx.getSourceManager(), ctx.getLangOpts())
    ,_astVisitor(nullptr)
    ,_attrManager(AttributeManager::instance())
{}


clang::Rewriter& SessionStage::getRewriter() {
  return _rewriter;
}

void SessionStage::setAstVisitor(ASTVisitor *visitor) {
  _astVisitor = visitor;
}

ASTVisitor *SessionStage::getVisitor() {
  assert(_astVisitor == nullptr);
  return _astVisitor;
}

TRANSPILER_TYPE SessionStage::getBackend() const {
  return _globalSession.targetBackend;
}

AttributeManager &SessionStage::getAttrManager() {
  return _attrManager;
}

const AttributeManager &SessionStage::getAttrManager() const {
  return _attrManager;
}

void SessionStage::writeTranspiledSource() {
  SourceManager &sm = _ctx.getSourceManager();
  const RewriteBuffer* rb = _rewriter.getRewriteBufferFor(sm.getMainFileID());
  if (!rb) {
    return;
  }
  std::string modifiedCode = std::string(rb->begin(), rb->end());
  _globalSession.transpiledCode = format(modifiedCode);
}

void SessionStage::setUserCtx(std::any userCtx) {
  _userCtx = userCtx;
}

std::any &SessionStage::getUserCtx() {
  return _userCtx;
}

const std::any &SessionStage::getUserCtx() const {
  return _userCtx;
}

}
