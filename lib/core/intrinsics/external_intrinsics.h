#pragma once
#include <clang/Basic/FileEntry.h>
#include <string>

namespace oklt {

class SessionStage;
class HeaderDepsInfo;

bool overrideExternalIntrinsic(SessionStage& stage,
                               HeaderDepsInfo& deps,
                               const std::string& includedFileName,
                               clang::OptionalFileEntryRef includedFile);

void updateExternalIntrinsicMap(SessionStage& stage, HeaderDepsInfo& deps);
}  // namespace oklt
