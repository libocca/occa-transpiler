#include "oklt/core/transpiler_session/transpiler_session.h"
#include "oklt/core/diag/diag_consumer.h"
#include "oklt/core/utils/format.h"

#include <clang/Basic/SourceManager.h>
#include <clang/AST/ParentMapContext.h>

namespace oklt {
using namespace clang;

TranspilerSession::TranspilerSession(TRANSPILER_TYPE backend)
    : targetBackend(backend), transpiledCode() {}

SessionStage::SessionStage(TranspilerSession& session, CompilerInstance& compiler)
    : _session(session),
      _compiler(compiler),
      _rewriter(_compiler.getSourceManager(), _compiler.getLangOpts()),
      _attrStore(_compiler.getASTContext()) {}

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

void SessionStage::pushDiagnosticMessage(clang::StoredDiagnostic &&message) {
  _diagMessages.emplace_back(message);
}

bool SessionStage::setUserCtx(const std::string& key, std::any userCtx) {
    auto it = _userCtxMap.find(key);
    if (it != _userCtxMap.end()) {
        return false;
    }

    _userCtxMap.insert({key, std::move(userCtx)});
    return true;
}

std::any SessionStage::getUserCtx(const std::string& key) {
    auto it = _userCtxMap.find(key);
    if (it == _userCtxMap.end()) {
        return std::any{};
    }
    return it->second;
}

SessionStage& getStageFromASTContext(clang::ASTContext& ast) {
  // NOTE:
  // There are a few stable references/pointer that can point to our controlled classes and structures.
  // getSourceManager().getFileManager -- Reference to FileManager. Can exist only one.
  // getDiagnostics().getClient() -- Pointer to DiagnosticConsumer. Multiplex.

  auto diag = dynamic_cast<DiagConsumer *>(ast.getDiagnostics().getClient());
  assert(diag);

  return diag->getSession();
}

}  // namespace oklt
