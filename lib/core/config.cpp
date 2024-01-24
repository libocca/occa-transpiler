#include <oklt/core/config.h>
#include <oklt/util/string_utils.h>
#include <algorithm>
#include <map>

namespace oklt {

tl::expected<TRANSPILER_TYPE, std::string> backendFromString(const std::string &type)
{
    static const std::map<std::string,  TRANSPILER_TYPE> BACKENDS_MAP = {
        {"cuda", TRANSPILER_TYPE::CUDA},
        {"openmp", TRANSPILER_TYPE::OPENMP}
    };

    auto it = BACKENDS_MAP.find(util::toLower(type));
    if(it != BACKENDS_MAP.end()) {
        return it->second;
    }
    return tl::unexpected("unknown backend is requisted");
}
}
