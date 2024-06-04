#pragma once
#include <clang/Basic/FileEntry.h>
#include <string>

namespace clang {
class CompilerInstance;
class SourceManager;
}  // namespace clang

namespace oklt {

class TranspilerSession;

bool overrideExternalIntrinsic(TranspilerSession& session,
                               const std::string& includedFileName,
                               clang::OptionalFileEntryRef includedFile,
                               clang::SourceManager& sourceManager);

}  // namespace oklt
