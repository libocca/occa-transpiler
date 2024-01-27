#include "oklt/core/transpiler_session/transpiler_session.h"
#include "oklt/core/diag/diag_consumer.h"
#include "oklt/core/utils/format.h"

#include <clang/AST/ParentMapContext.h>
#include <clang/Basic/SourceManager.h>

namespace oklt {
using namespace clang;

TranspilerSession::TranspilerSession(TRANSPILER_TYPE backend)
    : targetBackend(backend), transpiledCode() {}

SessionStage::SessionStage(TranspilerSession& session, CompilerInstance& compiler)
    : _session(session),
      _compiler(compiler),
      _rewriter(_compiler.getSourceManager(), _compiler.getLangOpts()) {}

clang::CompilerInstance& SessionStage::getCompiler() {
  return _compiler;
}

clang::Rewriter& SessionStage::getRewriter() {
  return _rewriter;
}

AttributeManager& SessionStage::getAttrManager() {
  return AttributeManager::instance();
}

std::string SessionStage::getRewriterResult() {
  auto* rewriteBuf = _rewriter.getRewriteBufferFor(_compiler.getSourceManager().getMainFileID());
  if (!rewriteBuf || rewriteBuf->size() == 0) {
    return "";
  }

  return std::string{rewriteBuf->begin(), rewriteBuf->end()};
}

TRANSPILER_TYPE SessionStage::getBackend() const {
  return _session.targetBackend;
}

void SessionStage::pushDiagnosticMessage(clang::StoredDiagnostic& message) {
  // TODO: Fixup sourceLocation
  auto msg = message.getMessage();
  auto lineNo = message.getLocation().getLineNumber();

  std::stringstream ss;
  ss << "line " << lineNo << ": ";
  ss << msg.str();

  _session.diagMessages.emplace_back(Error{ss.str()});
}

SessionStage* getStageFromASTContext(clang::ASTContext& ast) {
  // NOTE:
  // There are a few stable references/pointer that can point to our controlled classes and
  // structures. getSourceManager().getFileManager -- Reference to FileManager. Initialized before
  // CompilerInstance, exist only one. getDiagnostics().getClient() -- Pointer to
  // DiagnosticConsumer. Initialized during ExecuteAction, can be multiplexed.

  auto diag = dynamic_cast<DiagConsumer*>(ast.getDiagnostics().getClient());
  if (!diag) {
    return nullptr;
  }

  return &diag->getSession();
}

}  // namespace oklt
