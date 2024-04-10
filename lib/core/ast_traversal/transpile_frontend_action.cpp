#include "core/ast_traversal/transpile_frontend_action.h"
#include "core/ast_traversal/transpile_ast_consumer.h"
#include "core/diag/diag_consumer.h"
#include "core/transpiler_session/header_info.h"
#include "core/transpiler_session/session_stage.h"
#include "core/vfs/overlay_fs.h"

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

    // setup preprocessor hook to gather all user/system includes
    if (compiler.hasPreprocessor()) {
        auto& deps = _stage->tryEmplaceUserCtx<HeaderDepsInfo>();
        std::unique_ptr<PPCallbacks> callback =
            std::make_unique<InclusionDirectiveCallback>(deps, compiler.getSourceManager());

        compiler.getPreprocessor().addPPCallbacks(std::move(callback));
    }
    compiler.getDiagnostics().setShowColors(true);

    return std::move(astConsumer);
}

bool TranspileFrontendAction::PrepareToExecuteAction(CompilerInstance& compiler) {
    // set overlay fs to force compiler to take transpiled files
    if (compiler.hasFileManager()) {
        auto overlayFs = makeOverlayFs(compiler.getFileManager().getVirtualFileSystemPtr(),
                                       _session.normalizedHeaders);
        compiler.getFileManager().setVirtualFileSystem(overlayFs);
    }

    return true;
}

}  // namespace oklt
