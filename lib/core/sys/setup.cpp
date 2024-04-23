#include "core/sys/setup.h"

#include <clang/Basic/Version.h>
#include <spdlog/fmt/fmt.h>
#include <spdlog/spdlog.h>

#include <cstdlib>

namespace {
std::unique_ptr<std::string> CLANG_SYSROOT_DIR;
std::unique_ptr<std::string> CLANG_ISYSTEM_OPT;

constexpr char EB_ROOT_CLANG[] = "EBROOTCLANG";

__attribute__((constructor)) void initSysRoot() {
    CLANG_SYSROOT_DIR = std::make_unique<std::string>();
    CLANG_ISYSTEM_OPT = std::make_unique<std::string>();
    auto sysRoot = std::getenv(EB_ROOT_CLANG);
    if (sysRoot) {
        *CLANG_SYSROOT_DIR = sysRoot;
        *CLANG_ISYSTEM_OPT =
            fmt::format("isystem{}/lib/clang/{}/include", *CLANG_SYSROOT_DIR, CLANG_VERSION_MAJOR);
        SPDLOG_DEBUG("set additional clang opt: {}", *CLANG_ISYSTEM_OPT);
    }
}

}  // namespace

namespace oklt {
std::string getSysRoot() {
    return *CLANG_SYSROOT_DIR;
}

std::string getSysRootOpt() {
    return *CLANG_ISYSTEM_OPT;
}

}  // namespace oklt
