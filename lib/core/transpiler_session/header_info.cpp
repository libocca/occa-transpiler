#include "core/transpiler_session/header_info.h"
#include "core/intrinsics/builtin_intrinsics.h"
#include "core/intrinsics/external_intrinsics.h"
#include "core/transpiler_session/session_stage.h"

namespace {}
namespace oklt {

using namespace llvm;

InclusionDirectiveCallback::InclusionDirectiveCallback(SessionStage& stage,
                                                       HeaderDepsInfo& deps_,
                                                       clang::SourceManager& sm_)
    : deps(deps_),
      sm(sm_),
      _stage(stage) {}

void InclusionDirectiveCallback::InclusionDirective(clang::SourceLocation hashLoc,
                                                    const clang::Token& includeTok,
                                                    clang::StringRef fileName,
                                                    bool isAngled,
                                                    clang::CharSourceRange filenameRange,
                                                    clang::OptionalFileEntryRef file,
                                                    clang::StringRef searchPath,
                                                    clang::StringRef relativePath,
                                                    const clang::Module* imported,
                                                    clang::SrcMgr::CharacteristicKind fileType) {
    if (!deps.useOklIntrinsic) {
        deps.useOklIntrinsic = fileName == INTRINSIC_INCLUDE_FILENAME;
    }

    auto currentFileType = sm.getFileCharacteristic(hashLoc);
    if (clang::SrcMgr::isSystem(currentFileType)) {
        return;
    }

    auto fileNameStr = fileName.str();
    bool isIntrinsic = overrideExternalIntrinsic(_stage, deps, fileNameStr, file);

    auto dep = HeaderDep{
        .hashLoc = hashLoc,
        .includeTok = includeTok,
        .fileName = fileNameStr,
        .isAngled = isAngled,
        .filenameRange = filenameRange,
        .file = file,
        .searchPath = searchPath.str(),
        .relativePath = relativePath.str(),
        .imported = imported,
        .fileType = fileType,
    };

    if (isIntrinsic) {
        deps.externalIntrinsicHeaders.push_back(std::move(dep));
    } else {
        deps.topLevelDeps.push_back(std::move(dep));
    }
}
}  // namespace oklt
