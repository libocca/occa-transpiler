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
    }
    return {};
}
}  // namespace oklt
