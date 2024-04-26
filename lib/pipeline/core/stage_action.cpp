#include "core/transpiler_session/session_stage.h"
#include "core/vfs/overlay_fs.h"

#include "pipeline/core/stage_action.h"

#include <clang/Frontend/CompilerInstance.h>

#include <spdlog/spdlog.h>

namespace oklt {
using namespace clang;
/**
 * @brief Base class for run a stage of transpiler pipeline
 */
bool StageAction::PrepareToExecuteAction(clang::CompilerInstance& compiler) {
    // create stage
    _stage = std::make_unique<SessionStage>(*_session, compiler, getRewriterType());

    //
    if (!compiler.hasFileManager()) {
        SPDLOG_ERROR("no file manager at call of {}", __FUNCTION__);
        return false;
    }

    const auto& headers = _session->getStagedHeaders();
    if (!headers.empty()) {
        auto& fm = compiler.getFileManager();
        auto vfs = fm.getVirtualFileSystemPtr();

        auto overlayFs = makeOverlayFs(vfs, headers);
        fm.setVirtualFileSystem(overlayFs);
    }

    return true;
}

void StageAction::EndSourceFileAction() {
    if (!_session->getErrors().empty()) {
        return;
    }

    // set input for the next stage
    // copy transformed or original main source file
    auto& input = _session->getInput();
    auto& output = _session->getOutput();
    output.normalized.source = _stage->getRewriterResultForMainFile();

    // copy transformed headers and merge untouched headers for the next stage
    auto transformedHeaders = _stage->getRewriterResultForHeaders();
    transformedHeaders.fileMap.merge(_session->getStagedHeaders());
    output.normalized.headers = std::move(transformedHeaders.fileMap);
}

bool StageAction::setSession(SharedTranspilerSession session) {
    _session = std::move(session);
    return true;
}

}  // namespace oklt
