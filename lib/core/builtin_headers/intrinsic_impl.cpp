#include "core/builtin_headers/intrinsic_impl.h"
#include "core/builtin_headers/okl_intrinsic_cuda.h"
#include "core/builtin_headers/okl_intrinsic_dpcpp.h"
#include "core/builtin_headers/okl_intrinsic_hip.h"
#include "core/builtin_headers/okl_intrinsic_host.h"

#include "core/transpiler_session/transpiler_session.h"
#include "core/vfs/overlay_fs.h"
#include <clang/Frontend/CompilerInstance.h>

namespace oklt {

struct IntrinsicInfo {
    std::string source;
    std::vector<std::string> includes;
};

IntrinsicInfo getIntrinsicInfo(TargetBackend backend) {
    switch (backend) {
        case TargetBackend::CUDA:
            return IntrinsicInfo {
                std::string(INTRINSIC_CUDA),
                {"cuda_pipeline_primitives.h"}
            };
        case TargetBackend::DPCPP:
            return IntrinsicInfo {
                std::string(INTRINSIC_DPCPP),
                {}
            };
        case TargetBackend::HIP:
            return IntrinsicInfo {
                std::string(INTRINSIC_HIP),
                {}
            };
        case TargetBackend::OPENMP:
        case TargetBackend::SERIAL:
        case TargetBackend::_LAUNCHER:
            return IntrinsicInfo {
                std::string(INTRINSIC_HOST),
                {
                 "stddef.h",
                 "cmath",
                 "cstring",
                }
            };
        default:
            return {{}, {}};
    }
}

void addInstrinsicStub(TranspilerSession &session,
                       clang::CompilerInstance &compiler)
{
    auto& headers = session.getStagedHeaders();
    headers.emplace(INTRINSIC_INCLUDE_FILENAME,std::string());
    auto& fm = compiler.getFileManager();
    auto vfs = fm.getVirtualFileSystemPtr();

    auto overlayFs = makeOverlayFs(vfs, headers);
    fm.setVirtualFileSystem(overlayFs);
}

std::vector<std::string> embedInstrinsic(std::string &input,
                                         TargetBackend backend)
{
    auto info = getIntrinsicInfo(backend);
    input.insert(0, info.source);
    return info.includes;
}


}  // namespace oklt
