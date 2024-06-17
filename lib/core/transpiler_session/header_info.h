#pragma once

#include <clang/Basic/LLVM.h>
#include <clang/Lex/Preprocessor.h>

#include <map>
#include <string>

namespace oklt {
struct TransformedFiles {
    // name to file content map
    std::map<std::string, std::string> fileMap;
};
// Stub to collect data from InclusionDirective callbacks.
struct HeaderDep {
    clang::SourceLocation hashLoc;
    clang::Token includeTok;
    std::string fileName;
    bool isAngled;
    clang::CharSourceRange filenameRange;
    clang::OptionalFileEntryRef file;
    std::string searchPath;
    std::string relativePath;
    const clang::Module* imported;
    clang::SrcMgr::CharacteristicKind fileType;
};
using HeaderIncStack = std::vector<HeaderDep>;

struct HeaderDepsInfo {
    std::vector<HeaderDep> topLevelDeps;
    std::vector<std::string> backendHeaders;
    std::vector<std::string> backendNss;
    std::vector<HeaderDep> externalIntrinsicHeaders;
    std::map<std::string, std::string> externalIntrinsicsSources;
    bool useOklIntrinsic = false;
};

class SessionStage;

class InclusionDirectiveCallback : public clang::PPCallbacks {
   public:
    InclusionDirectiveCallback(SessionStage& session,
                               HeaderDepsInfo& depsInfo,
                               clang::SourceManager& sm);
    void InclusionDirective(clang::SourceLocation HashLoc,
                            const clang::Token& IncludeTok,
                            clang::StringRef fileName,
                            bool IsAngled,
                            clang::CharSourceRange FilenameRange,
                            clang::OptionalFileEntryRef File,
                            clang::StringRef SearchPath,
                            clang::StringRef RelativePath,
                            const clang::Module* Imported,
                            clang::SrcMgr::CharacteristicKind FileType) override;

   private:
    HeaderDepsInfo& deps;
    clang::SourceManager& sm;
    SessionStage& _stage;
};

}  // namespace oklt
