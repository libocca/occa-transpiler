#include <clang/Basic/SourceManager.h>

#include "oklt/core/transpile_session/transpile_session.h"
#include "oklt/core/utils/format.h"


namespace oklt {
using namespace clang;

TranspileSession::TranspileSession(const TranspilerConfig &config,
                                   ASTContext &ctx)
    :_ctx(ctx)
    ,_transpileConfig(config)
    ,_rewriter(ctx.getSourceManager(), ctx.getLangOpts())
    ,_errorReporter()
    ,_astVisitor(nullptr)
    ,_attrManager(AttributeManager::instance())
{}


clang::Rewriter& TranspileSession::getRewriter() {
  return _rewriter;
}

void TranspileSession::setAstVisitor(ASTVisitor *visitor) {
  _astVisitor = visitor;
}

ASTVisitor *TranspileSession::getVisitor() {
  assert(_astVisitor == nullptr);
  return _astVisitor;
}

TRANSPILER_TYPE TranspileSession::getBackend() const {
  return _transpileConfig.backendType;
}

AttributeManager &TranspileSession::getAttrManager() {
  return _attrManager;
}

const AttributeManager &TranspileSession::getAttrManager() const {
  return _attrManager;
}

ErrorReporter &TranspileSession::getErrorReporter() {
  return _errorReporter;
}

void TranspileSession::writeTranspiledSource() {
  SourceManager &sm = _ctx.getSourceManager();
  const RewriteBuffer* rb = _rewriter.getRewriteBufferFor(sm.getMainFileID());
  if (!rb) {
    return;
  }
  std::string modifiedCode = std::string(rb->begin(), rb->end());
  // INFO: make it pretty
  std::string formated = format(modifiedCode);
  _transpileConfig.transpiledOutput << formated;
}

}
