#pragma once
#include <string>
#include <clang/Basic/FileEntry.h>

namespace clang {
class CompilerInstance;
class SourceManager;
}

namespace oklt {

class TranspilerSession;

//bool isExternalInstrincisInclude(TranspilerSession &session,
//                                 const std::string &fileName);

void overrideExternalIntrinsic(TranspilerSession &session,
                               const std::string &includedFileName,
                               clang::OptionalFileEntryRef includedFile,
                               clang::SourceManager &sourceManager);

}  // namespace oklt
