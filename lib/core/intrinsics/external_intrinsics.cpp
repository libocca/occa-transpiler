#include "core/intrinsics/external_intrinsics.h"

#include <clang/Frontend/CompilerInstance.h>
#include "core/transpiler_session/transpiler_session.h"
#include "util/string_utils.hpp"

#include <algorithm>

namespace oklt {

using namespace llvm;
namespace fs = std::filesystem;

tl::expected<fs::path, std::string>
getIntrincisImplSourcePath(TargetBackend backend,
                       const fs::path &intrincisPath)
{
    switch (backend) {
        case TargetBackend::CUDA:
            return intrincisPath / "cuda";
        case TargetBackend::HIP:
            return intrincisPath / "hip";
        case TargetBackend::DPCPP:
            return intrincisPath / "dpcpp";
        case TargetBackend::OPENMP:
            return intrincisPath / "openmp";
        case TargetBackend::SERIAL:
            return intrincisPath / "serial";
        default:
            return tl::make_unexpected("User intrinsic does not implement target backend");
     }
}

bool isExternalInstrincisInclude(TranspilerSession &session,
                                 const std::string &fileName)
{
     auto maybeUserIntrinsic = session.getInput().userIntrincis;
     if(!maybeUserIntrinsic) {
        return false;
     }
     auto instrinsic = maybeUserIntrinsic.value();
     auto folderPrefix = instrinsic.filename().string();
     return util::starts_with(fileName, folderPrefix);
}


tl::expected<std::unique_ptr<llvm::MemoryBuffer>, std::string>
getExternalIntrinsicSource(TargetBackend backend,
                           const fs::path &intrinsicPath,
                           clang::SourceManager& sm)
{

     auto implPathResult = getIntrincisImplSourcePath(backend, intrinsicPath);
     if(!implPathResult) {
        return tl::make_unexpected(implPathResult.error());
     }

     auto sourceFolder = implPathResult.value();
     if(!std::filesystem::exists(sourceFolder)) {
        return tl::make_unexpected("Intrinsic implementation folder does not exist");
     }

     std::vector<fs::path> files(fs::directory_iterator(sourceFolder), {});
     if(files.empty()) {
        return tl::make_unexpected("Intrinsic implementation files is missing");
     }

     auto it = std::find_if(files.cbegin(), files.cend(), [](const fs::path &p) -> bool {
         return p.extension().string() == std::string(".h");
     });

     if(it == files.cend()) {
        std::string error = "Can't' find implementation file with path: " + sourceFolder.string();
        return tl::make_unexpected(error);
     }

     auto &fm = sm.getFileManager();
     auto backendFilePath = it->string();
     auto maybeReplacedFile = fm.getFile(backendFilePath);
     if(!maybeReplacedFile) {
        std::string error = "Can't open file: " + backendFilePath;
        return tl::make_unexpected(error);
     }
     auto maybeBuffer = fm.getBufferForFile(maybeReplacedFile.get());
     if (!maybeBuffer) {
        std::string error = "Can't get memory buffer for: " + backendFilePath;
        return tl::make_unexpected(error);
     }
     return std::move(maybeBuffer.get());
}

void overrideExternalIntrinsic(TranspilerSession &session,
                               const std::string &includedFileName,
                               clang::OptionalFileEntryRef includedFile,
                               clang::SourceManager &sourceManager)
{
     auto maybeIntrinsicPath = session.getInput().userIntrincis;
     if(maybeIntrinsicPath && isExternalInstrincisInclude(session, includedFileName)) {
        auto intrinsicPath = maybeIntrinsicPath.value();
        auto infoResult = getExternalIntrinsicSource(session.getInput().backend, intrinsicPath, sourceManager);
        if(!infoResult) {
            session.pushError(std::error_code(), infoResult.error());
            return;
        }
        auto buffer = std::move(infoResult.value());
        auto &fm = sourceManager.getFileManager();
        if(includedFile) {
            auto fileRef = includedFile;
            const auto &fileEntry = fileRef->getFileEntry();
            sourceManager.overrideFileContents(&fileEntry, std::move(buffer));
        } else {
            //INFO: case when the file can be found by relative path
            //      it happens when the include path is relative to WORKING DIR path
            auto maybeFileRef = fm.getFileRef(includedFileName);
            if(maybeFileRef) {
                auto foundFileRef = maybeFileRef.get();
                sourceManager.overrideFileContents(foundFileRef, std::move(buffer));
            }
        }
     }
}

}  // namespace oklt
