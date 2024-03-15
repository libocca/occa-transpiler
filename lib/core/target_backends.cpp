#include <oklt/core/target_backends.h>
#include <oklt/util/string_utils.h>

#include <map>

namespace oklt {

tl::expected<TargetBackend, std::string> backendFromString(const std::string& type) {
    static const std::map<std::string, TargetBackend> BACKENDS_MAP = {
        {"serial", TargetBackend::SERIAL},
        {"openmp", TargetBackend::OPENMP},
        {"cuda", TargetBackend::CUDA},
        {"hip", TargetBackend::HIP},
        {"dpcpp", TargetBackend::DPCPP},
        {"launcher", TargetBackend::_LAUNCHER},
    };

    auto it = BACKENDS_MAP.find(util::toLower(type));
    if (it != BACKENDS_MAP.end()) {
        return it->second;
    }
    return tl::unexpected("unknown backend is requested");
}

std::string backendToString(TargetBackend backend) {
    switch (backend) {
        case TargetBackend::SERIAL:
            return std::string{"serial"};
        case TargetBackend::OPENMP:
            return std::string{"openmp"};
        case TargetBackend::CUDA:
            return std::string{"cuda"};
        case TargetBackend::HIP:
            return std::string{"hip"};
        case TargetBackend::DPCPP:
            return std::string{"dpcpp"};
        case TargetBackend::_LAUNCHER:
            return std::string{"launcher"};
    }
    return {};
}

bool isHostCategory(TargetBackend backend) {
    switch (backend) {
        case TargetBackend::SERIAL:
        case TargetBackend::OPENMP:
            return true;
        default:
            return false;
    }
}

bool isDeviceCategory(TargetBackend backend) {
    switch (backend) {
        case TargetBackend::CUDA:
        case TargetBackend::HIP:
        case TargetBackend::DPCPP:
            return true;
        default:
            return false;
    }
}

}  // namespace oklt
