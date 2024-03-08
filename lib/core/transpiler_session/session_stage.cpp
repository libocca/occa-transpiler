#include "core/transpiler_session/session_stage.h"
#include "core/attribute_manager/attribute_manager.h"
#include "core/diag/diag_consumer.h"
#include "core/transpiler_session/transpiler_session.h"

#include <clang/AST/ParentMapContext.h>
#include <clang/Basic/SourceManager.h>

namespace oklt {
using namespace clang;

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

std::string SessionStage::getRewriterResultOfMainFile() {
    auto* rewriteBuf = _rewriter.getRewriteBufferFor(_compiler.getSourceManager().getMainFileID());
    if (!rewriteBuf || rewriteBuf->size() == 0) {
        return "";
    }

    return std::string{rewriteBuf->begin(), rewriteBuf->end()};
}

std::map<std::string, std::string> SessionStage::getAllRewriterResults() {
    std::map<std::string, std::string> allChangedFiles;
    for (auto it = _rewriter.buffer_begin(); it != _rewriter.buffer_end(); ++it) {
        std::string fileName =
            _compiler.getSourceManager().getFileEntryForID(it->first)->getName().data();
        allChangedFiles[fileName] = [](const auto& buf) -> std::string {
            if (buf.size() == 0) {
                return "";
            }

            return std::string{buf.begin(), buf.end()};
        }(it->second);

        llvm::outs() << "overlayFs file: " << fileName << "\n"
                     << "source:\n"
                     << allChangedFiles.at(fileName) << '\n';
    }

    return allChangedFiles;
}

TargetBackend SessionStage::getBackend() const {
    return _session.input.backend;
}

AstProcessorType SessionStage::getAstProccesorType() const {
    return _session.input.astProcType;
}

void SessionStage::pushDiagnosticMessage(clang::StoredDiagnostic& message) {
    _session.pushDiagnosticMessage(message);
}

void SessionStage::pushError(std::error_code ec, std::string desc) {
    _session.pushError(ec, std::move(desc));
}

void SessionStage::pushError(const Error& err) {
    _session.pushError(err.ec, std::move(err.desc));
}

void SessionStage::pushWarning(std::string desc) {
    _session.pushWarning(std::move(desc));
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
