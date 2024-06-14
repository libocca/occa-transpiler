#pragma once
#include <clang/Basic/FileEntry.h>
#include <string>

namespace clang {
class CompilerInstance;
class SourceManager;
}  // namespace clang

namespace oklt {

class TranspilerSession;
class TransformedFiles;
class SessionStage;
class HeaderDepsInfo;

bool overrideExternalIntrinsic(TranspilerSession& session,
                               const std::string& includedFileName,
                               clang::OptionalFileEntryRef includedFile,
                               clang::SourceManager& sourceManager);

void nullyLauncherExternalIntrinsics(TransformedFiles& inputs, SessionStage& stage);

void embedLauncherExternalIntrinsics(std::string& input,
                                     const HeaderDepsInfo& info,
                                     SessionStage& stage);

}  // namespace oklt
