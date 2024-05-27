#include "core/intrinsics/builtin_intrinsics.h"
#include "core/intrinsics/okl_intrinsic_cuda.h"
#include "core/intrinsics/okl_intrinsic_dpcpp.h"
#include "core/intrinsics/okl_intrinsic_hip.h"
#include "core/intrinsics/okl_intrinsic_host.h"

#include <clang/Frontend/CompilerInstance.h>
#include "core/transpiler_session/transpiler_session.h"
#include "core/vfs/overlay_fs.h"

namespace oklt {

using namespace llvm;

struct IntrinsicInfo {
    std::string source;
    std::vector<std::string> includes;
};

IntrinsicInfo getIntrinsicInfo(TargetBackend backend) {
    switch (backend) {
        case TargetBackend::CUDA:
            return IntrinsicInfo{std::string(INTRINSIC_CUDA), {"cuda_pipeline_primitives.h"}};
        case TargetBackend::DPCPP:
            return IntrinsicInfo{std::string(INTRINSIC_DPCPP), {}};
        case TargetBackend::HIP:
            return IntrinsicInfo{std::string(INTRINSIC_HIP), {}};
        case TargetBackend::OPENMP:
        case TargetBackend::SERIAL:
        case TargetBackend::_LAUNCHER:
            return IntrinsicInfo{std::string(INTRINSIC_HOST),
                                 {
                                     "stddef.h",
                                     "cmath",
                                     "cstring",
                                 }};
        default:
            return {{}, {}};
    }
}

void addInstrinsicStub(TranspilerSession& session, clang::CompilerInstance& compiler) {
    auto& headers = session.getStagedHeaders();
    headers.emplace(INTRINSIC_INCLUDE_FILENAME, std::string());
    auto& fm = compiler.getFileManager();
    auto vfs = fm.getVirtualFileSystemPtr();
    if (vfs) {
        vfs::OverlayFileSystem* ptr = dynamic_cast<vfs::OverlayFileSystem*>(vfs.get());

        for (const auto& elem : headers) {
            for (auto& overlay : ptr->overlays_range()) {
                vfs::InMemoryFileSystem* inMemory =
                    dynamic_cast<vfs::InMemoryFileSystem*>(overlay.get());
                if (inMemory && !inMemory->exists(elem.first)) {
                    inMemory->addFile(elem.first, 0, MemoryBuffer::getMemBuffer(elem.second));
                }
            }
        }
    } else {
        auto overlayFs = makeOverlayFs(vfs, headers);
        fm.setVirtualFileSystem(overlayFs);
    }
}

std::vector<std::string> embedInstrinsic(std::string& input, TargetBackend backend) {
    auto info = getIntrinsicInfo(backend);
    input.insert(0, info.source);
    return info.includes;
}

}  // namespace oklt
