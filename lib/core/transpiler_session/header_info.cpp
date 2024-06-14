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
      _session(session),
      _isInExternalIntrinsic(false) {}

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
    auto isIntrinsic = overrideExternalIntrinsic(_session, fileNameStr, file, sm);
    if (isIntrinsic) {
        deps.externalIntrinsics.push_back(fileNameStr);
        _extIntrinsicFID = sm.getFileID(hashLoc);
    }

    // INFO: force instrinsic includes to be system includes
    //       they will be removed & attached to final transpiled source
    if (_isInExternalIntrinsic) {
        fileType = clang::SrcMgr::CharacteristicKind::C_System;
        deps.externalIntrinsicDeps.push_back(HeaderDep{
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
    } else {
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
}

void InclusionDirectiveCallback::FileChanged(clang::SourceLocation Loc,
                                             FileChangeReason Reason,
                                             clang::SrcMgr::CharacteristicKind FileType,
                                             clang::FileID PrevFID) {
    if (Reason == FileChangeReason::EnterFile) {
        if (_extIntrinsicFID.isValid()) {
            _isInExternalIntrinsic = PrevFID == _extIntrinsicFID;
            return;
        }
    }

    if (Reason == FileChangeReason::ExitFile) {
        auto thisFID = sm.getFileID(Loc);
        if (_extIntrinsicFID.isValid() && thisFID == _extIntrinsicFID) {
            _extIntrinsicFID = clang::FileID();
            _isInExternalIntrinsic = false;
            return;
        }
    }
}

bool InclusionDirectiveCallback::FileNotFound(clang::StringRef FileName) {
    return _isInExternalIntrinsic;
}

}  // namespace oklt
