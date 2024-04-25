#include "core/builtin_headers/intrinsic_impl.h"
#include "core/builtin_headers/okl_intrinsic_cuda.h"
#include "core/builtin_headers/okl_intrinsic_dpcpp.h"
#include "core/builtin_headers/okl_intrinsic_hip.h"
#include "core/builtin_headers/okl_intrinsic_openmp.h"

namespace oklt {

std::string getIntrinsicIncSource(TargetBackend backend) {
    switch (backend) {
    case TargetBackend::CUDA:
        return std::string(INTRINSIC_CUDA);
    case TargetBackend::DPCPP:
        return std::string(INTRINSIC_DPCPP);
    case TargetBackend::HIP:
        return std::string(INTRINSIC_HIP);
    case TargetBackend::OPENMP:
        return std::string(INTRINSIC_OPENMP);
    case TargetBackend::SERIAL:
        return {};
    default:
        return {};
    }
}

}
