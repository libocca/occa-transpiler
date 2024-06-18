#include "core/intrinsics/external_intrinsics.h"

#include <clang/Frontend/CompilerInstance.h>
#include "core/transpiler_session/session_stage.h"
#include "core/transpiler_session/transpiler_session.h"
#include "oklt/util/io_helper.h"
#include "util/string_utils.hpp"

#include <algorithm>
#include <optional>

namespace oklt {

using namespace llvm;
namespace fs = std::filesystem;

tl::expected<fs::path, std::string> getIntrincisImplSourcePath(TargetBackend backend,
                                                               const fs::path& intrincisPath) {
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
        case TargetBackend::_LAUNCHER:
            return intrincisPath / "launcher";
        default:
            return tl::make_unexpected("User intrinsic does not implement target backend");
    }
}

std::string normalizedFileName(const std::string& fileName) {
    auto normalizedName = fileName;
    if (util::startsWith(normalizedName, "./")) {
        normalizedName = normalizedName.substr(2);
    }
    return normalizedName;
}

bool isExternalIntrinsicInclude(TranspilerSession& session, const std::string& fileName) {
    const auto& userIntrinsic = session.getInput().userIntrinsics;
    if (userIntrinsic.empty()) {
        return false;
    }
    auto normalizedName = normalizedFileName(fileName);
    for (const auto& intrinsic : userIntrinsic) {
        auto folderPrefix = intrinsic.filename().string();
        if (util::startsWith(normalizedName, folderPrefix)) {
            return true;
        }
    }
    return false;
}

std::optional<fs::path> getExternalInstrincisInclude(TranspilerSession& session,
                                                     const std::string& fileName) {
    const auto& userIntrinsic = session.getInput().userIntrinsics;
    if (userIntrinsic.empty()) {
        return std::nullopt;
    }

    auto normalizedName = normalizedFileName(fileName);
    for (const auto& intrinsic : userIntrinsic) {
        auto folderPrefix = intrinsic.filename().string();
        if (util::startsWith(normalizedName, folderPrefix)) {
            return intrinsic;
        }
    }
    return std::nullopt;
}

tl::expected<std::string, std::string> getExternalIntrinsicSource(TargetBackend backend,
                                                                  const fs::path& intrinsicPath,
                                                                  clang::SourceManager& sm) {
    auto implPathResult = getIntrincisImplSourcePath(backend, intrinsicPath);
    if (!implPathResult) {
        return tl::make_unexpected(implPathResult.error());
    }

    auto sourceFolder = implPathResult.value();
    if (!std::filesystem::exists(sourceFolder)) {
        return tl::make_unexpected("Intrinsic implementation folder does not exist");
    }

    std::vector<fs::path> files(fs::directory_iterator(sourceFolder), {});
    if (files.empty()) {
        return tl::make_unexpected("Intrinsic implementation files is missing");
    }

    auto it = std::find_if(files.cbegin(), files.cend(), [](const fs::path& p) -> bool {
        return p.extension().string() == std::string(".cpp");
    });

    if (it == files.cend()) {
        std::string error = "Can't' find implementation file with path: " + sourceFolder.string();
        return tl::make_unexpected(error);
    }

    auto contentResult = util::readFileAsStr(*it);
    if (!contentResult) {
        std::string error = "Can't get memory buffer for: " + it->string();
        return tl::make_unexpected(error);
    }
    return contentResult.value();
}

bool overrideExternalIntrinsic(SessionStage& stage,
                               HeaderDepsInfo& deps,
                               const std::string& includedFileName,
                               clang::OptionalFileEntryRef includedFile) {
    auto& session = stage.getSession();
    auto& sourceManager = stage.getCompiler().getSourceManager();
    const auto& userIntrinsics = session.getInput().userIntrinsics;
    if (!userIntrinsics.empty()) {
        auto maybeIntrinsicPath = getExternalInstrincisInclude(session, includedFileName);
        if (!maybeIntrinsicPath) {
            return false;
        }
        auto intrinsicPath = maybeIntrinsicPath.value();
        auto intrinsicResult =
            getExternalIntrinsicSource(stage.getBackend(), intrinsicPath, sourceManager);
        if (!intrinsicResult) {
            session.pushError(std::error_code(), intrinsicResult.error());
            return false;
        }
        deps.externalIntrinsicsSources[includedFileName] = std::move(intrinsicResult.value());

        auto emptyExternalIntrinsic = MemoryBuffer::getMemBuffer("");
        if (includedFile) {
            auto fileRef = includedFile;
            const auto& fileEntry = fileRef->getFileEntry();
            sourceManager.overrideFileContents(&fileEntry, std::move(emptyExternalIntrinsic));
        } else {
            // INFO: case when the file can be found by relative path
            //       it happens when the include path is relative to WORKING DIR path
            auto& fm = sourceManager.getFileManager();
            auto maybeFileRef = fm.getFileRef(includedFileName);
            if (maybeFileRef) {
                auto foundFileRef = maybeFileRef.get();
                sourceManager.overrideFileContents(foundFileRef, std::move(emptyExternalIntrinsic));
            }
        }
        return true;
    }
    return false;
}

void updateExternalIntrinsicMap(SessionStage& stage, HeaderDepsInfo& deps) {
    if (deps.externalIntrinsicsSources.empty()) {
        return;
    }

    auto backend = stage.getBackend();
    auto& session = stage.getSession();
    auto& sm = stage.getCompiler().getSourceManager();
    for (auto& mappedIntrinsic : deps.externalIntrinsicsSources) {
        auto maybeIntrinsicPath = getExternalInstrincisInclude(session, mappedIntrinsic.first);
        if (!maybeIntrinsicPath) {
            std::string error = "Count not find implementation for " + mappedIntrinsic.first;
            session.pushError(std::error_code(), error);
            return;
        }
        auto intrinsicPath = maybeIntrinsicPath.value();
        auto intrinsicResult = getExternalIntrinsicSource(backend, intrinsicPath, sm);
        if (!intrinsicResult) {
            session.pushError(std::error_code(), intrinsicResult.error());
            return;
        }
        mappedIntrinsic.second = std::move(intrinsicResult.value());
    }
}
}  // namespace oklt
