#include <spdlog/common.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <tl/expected.hpp>


namespace {

tl::expected<spdlog::level::level_enum, std::string> parseLoggerLevel(std::string levelStr) {
    std::transform(levelStr.begin(), levelStr.end(), levelStr.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    // TODO: magic_enum can hide these ifs
    if (levelStr == "trace") {
        return spdlog::level::trace;
    }

    if (levelStr == "debug") {
        return spdlog::level::debug;
    }

    if (levelStr == "info") {
        return spdlog::level::info;
    }

    if (levelStr == "warn") {
        return spdlog::level::warn;
    }

    if (levelStr == "err") {
        return spdlog::level::err;
    }

    if (levelStr == "critical") {
        return spdlog::level::critical;
    }
    return tl::make_unexpected("Failed to parse logger level name");
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
