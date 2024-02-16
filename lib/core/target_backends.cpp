#include <oklt/core/target_backends.h>
#include <oklt/util/string_utils.h>

#include <map>

namespace oklt {

tl::expected<TargetBackend, std::string> backendFromString(const std::string& type) {
    static const std::map<std::string, TargetBackend> BACKENDS_MAP = {
        {"cuda", TargetBackend::CUDA},
        {"openmp", TargetBackend::OPENMP},
        {"hip", TargetBackend::HIP},
    };

    auto it = BACKENDS_MAP.find(util::toLower(type));
    if (it != BACKENDS_MAP.end()) {
        return it->second;
    }
    return tl::unexpected("unknown backend is requisted");
}

std::string backendToString(TargetBackend backend) {
    switch (backend) {
        case TargetBackend::CUDA:
            return std::string{"cuda"};
        case TargetBackend::OPENMP:
            return std::string{"openmp"};
    }
    return {};
}
}  // namespace oklt
