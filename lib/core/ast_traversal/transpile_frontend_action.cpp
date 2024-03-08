#include "core/ast_traversal/transpile_frontend_action.h"
#include "core/ast_traversal/transpile_ast_consumer.h"
#include "core/diag/diag_consumer.h"
#include "core/transpiler_session/session_stage.h"

#include <memory>

namespace oklt {

using namespace clang;

TranspileFrontendAction::TranspileFrontendAction(TranspilerSession& session)
    : _session(session),
      _stage(nullptr) {}

std::unique_ptr<ASTConsumer> TranspileFrontendAction::CreateASTConsumer(CompilerInstance& compiler,
                                                                        llvm::StringRef in_file) {
    _stage = std::make_unique<SessionStage>(_session, compiler);
    auto astConsumer = std::make_unique<TranspileASTConsumer>(*_stage);
    compiler.getDiagnostics().setClient(new DiagConsumer(*_stage));
    return std::move(astConsumer);
}

bool TranspileFrontendAction::PrepareToExecuteAction(CompilerInstance& compiler) {
    if (compiler.hasFileManager()) {
        llvm::IntrusiveRefCntPtr<llvm::vfs::OverlayFileSystem> overlayFs(
            new llvm::vfs::OverlayFileSystem(llvm::vfs::getRealFileSystem()));
        llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> inMemoryFs(
            new llvm::vfs::InMemoryFileSystem);
        overlayFs->pushOverlay(inMemoryFs);

        for (const auto& f : _session.input.sourceCodes) {
            llvm::outs() << "transS overlayFs file: " << f.first << "\n"
                         << "source:\n"
                         << f.second << '\n';
            inMemoryFs->addFile(f.first, 0, llvm::MemoryBuffer::getMemBuffer(f.second));
        }

        compiler.getFileManager().setVirtualFileSystem(overlayFs);
    }

    return true;
}

}  // namespace oklt
