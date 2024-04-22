#include "core/sys/setup.h"
#include <spdlog/fmt/fmt.h>
#include <cstdlib>

namespace {
std::string CLANG_SYSROOT_DIR = "/";
std::string CLANG_SYSROOT_OPT;

constexpr char EB_ROOT_CLANG[] = "EBROOTCALNG";

__attribute__((constructor)) void initSysRoot() {
    auto sysRoot = std::getenv(EB_ROOT_CLANG);
    if (sysRoot) {
        CLANG_SYSROOT_DIR = sysRoot;
        CLANG_SYSROOT_OPT = fmt::format("--sysroot={}", CLANG_SYSROOT_DIR);
    }
}

}  // namespace

namespace oklt {
std::string getSysRoot() {
    return CLANG_SYSROOT_DIR;
}

std::string getSysRootOpt() {
    return CLANG_SYSROOT_OPT;
}
}  // namespace oklt
