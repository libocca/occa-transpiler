#include "core/transpiler_session/header_info.h"
#include "core/builtin_headers/intrinsic_impl.h"

namespace {}
namespace oklt {
InclusionDirectiveCallback::InclusionDirectiveCallback(HeaderDepsInfo& deps_,
                                                       const clang::SourceManager& sm_)
    : deps(deps_),
      sm(sm_) {}

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
    if(!deps.useOklIntrinsic) {
        deps.useOklIntrinsic = fileName == INTRINSIC_INCLUDE_FILENAME;
    }

    auto currentFileType = sm.getFileCharacteristic(hashLoc);
    if (clang::SrcMgr::isSystem(currentFileType)) {
        return;
    }

    deps.topLevelDeps.push_back(HeaderDep{
        .hashLoc = hashLoc,
        .includeTok = includeTok,
        .fileName = fileName.str(),
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
