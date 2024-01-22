#include "oklt/core/config.h"
#include <algorithm>
#include <map>

namespace oklt {


std::string toLower(const std::string &value) {
    std::string ret = value;
    std::transform(ret.begin(), ret.end(), ret.begin(), ::tolower);
    return ret;
}

tl::expected<TRANSPILER_TYPE, std::string> backendFromString(const std::string &type)
{
    static const std::map<std::string,  TRANSPILER_TYPE> BACKENDS_MAP = {
        {"cuda", TRANSPILER_TYPE::CUDA},
        {"openmp", TRANSPILER_TYPE::OPENMP}
    };

    auto it = BACKENDS_MAP.find(toLower(type));
    if(it != BACKENDS_MAP.end()) {
        return it->second;
    }
    return tl::unexpected("unknown backend is requisted");
}
}
