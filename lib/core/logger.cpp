#include <spdlog/common.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tl/expected.hpp>

#include <map>

namespace {

tl::expected<spdlog::level::level_enum, std::string> parseLoggerLevel(std::string levelStr) {
    std::transform(levelStr.begin(), levelStr.end(), levelStr.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    static std::map<std::string, spdlog::level::level_enum> mapping{
        {"trace", spdlog::level::trace},
        {"debug", spdlog::level::debug},
        {"info", spdlog::level::info},
        {"warn", spdlog::level::warn},
        {"err", spdlog::level::err},
        {"critical", spdlog::level::critical}};

    if (!mapping.count(levelStr)) {
        return tl::make_unexpected("Failed to parse logger level name");
    }
    return mapping.at(levelStr);
}

__attribute__((constructor)) void initLogger() {
    auto log_level_env = std::getenv("OKL_LOGGER_LEVEL");
    auto log_level = spdlog::level::info;  // Default: info level
    if (log_level_env) {
        auto log_level_status = parseLoggerLevel(log_level_env);
        if (log_level_status) {
            log_level = log_level_status.value();
        } else {
            spdlog::error(log_level_status.error());
        }
    }
    spdlog::set_level(log_level);
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%L%$] %v [%s:%#]");
    // spdlog::set_pattern("[%H:%M:%S.%e] [%^%L%$] %v [%@]");   // Absolute file path
}

}  // namespace
