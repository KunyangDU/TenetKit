#pragma once
// include/tenet/utils/logger.hpp
//
// Thin spdlog wrapper (header-only).
// Use TENET_LOG_INFO / TENET_LOG_DEBUG / … macros throughout the library.

#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>
#include <string>

namespace tenet {

inline std::shared_ptr<spdlog::logger>& _logger_instance() {
    static std::shared_ptr<spdlog::logger> inst;
    return inst;
}

// Call once in main() to configure the global logger.
inline void init_logger(const std::string& name  = "tenet",
                        spdlog::level::level_enum level = spdlog::level::info) {
    auto& inst = _logger_instance();
    if (!inst) {
        inst = spdlog::stdout_color_mt(name);
    }
    inst->set_level(level);
    spdlog::set_default_logger(inst);
}

inline std::shared_ptr<spdlog::logger> get_logger() {
    auto& inst = _logger_instance();
    if (!inst) {
        // Auto-initialise with defaults if init_logger was not called.
        init_logger();
    }
    return inst;
}

} // namespace tenet

#define TENET_LOG_TRACE(...)    tenet::get_logger()->trace(__VA_ARGS__)
#define TENET_LOG_DEBUG(...)    tenet::get_logger()->debug(__VA_ARGS__)
#define TENET_LOG_INFO(...)     tenet::get_logger()->info(__VA_ARGS__)
#define TENET_LOG_WARN(...)     tenet::get_logger()->warn(__VA_ARGS__)
#define TENET_LOG_ERROR(...)    tenet::get_logger()->error(__VA_ARGS__)
