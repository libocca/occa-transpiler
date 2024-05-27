#include "core/transpiler_session/header_info.h"
#include "core/intrinsics/builtin_intrinsics.h"
#include "core/intrinsics/external_intrinsics.h"
#include "core/transpiler_session/transpiler_session.h"

namespace {}
namespace oklt {

using namespace llvm;

InclusionDirectiveCallback::InclusionDirectiveCallback(TranspilerSession& session,
                                                       HeaderDepsInfo& deps_,
                                                       clang::SourceManager& sm_)
    : deps(deps_),
      sm(sm_),
      _session(session) {}

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
    overrideExternalIntrinsic(_session, fileNameStr, file, sm);

    deps.topLevelDeps.push_back(HeaderDep{
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

    });
}
}  // namespace oklt
