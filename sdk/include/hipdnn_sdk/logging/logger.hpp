// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "spdlog/sinks/stdout_color_sinks.h"
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

#ifdef ENABLE_BACKEND_LOGGING

#define HIPDNN_LOG_INFO(...)                                                \
    if(!hipdnn::logging::G_LOGGING_INITIALIZED)                             \
    {                                                                       \
        hipdnn::logging::initialize_logging_based_on_environment_variables( \
            hipdnn::logging::G_LOGGING_AREA);                               \
    }                                                                       \
    spdlog::info(__VA_ARGS__);

#define HIPDNN_LOG_WARN(...)                                                \
    if(!hipdnn::logging::G_LOGGING_INITIALIZED)                             \
    {                                                                       \
        hipdnn::logging::initialize_logging_based_on_environment_variables( \
            hipdnn::logging::G_LOGGING_AREA);                               \
    }                                                                       \
    spdlog::warn(__VA_ARGS__);

#define HIPDNN_LOG_ERROR(...)                                               \
    if(!hipdnn::logging::G_LOGGING_INITIALIZED)                             \
    {                                                                       \
        hipdnn::logging::initialize_logging_based_on_environment_variables( \
            hipdnn::logging::G_LOGGING_AREA);                               \
    }                                                                       \
    spdlog::error(__VA_ARGS__);

#define HIPDNN_LOG_INFO_WITH_HANDLE(handle, ...)                                \
    if(handle)                                                                  \
    {                                                                           \
        throw not_implemented_exception("handle logging not implemented yet."); \
    }
#else
#define HIPDNN_LOG_INFO(...) spdlog::info(__VA_ARGS__);
#define HIPDNN_LOG_WARN(...) spdlog::warn(__VA_ARGS__);
#define HIPDNN_LOG_ERROR(...) spdlog::error(__VA_ARGS__);
#define HIPDNN_LOG_INFO_WITH_HANDLE(handle, ...)
#endif

namespace hipdnn::logging
{
#ifdef ENABLE_BACKEND_LOGGING
inline bool G_LOGGING_INITIALIZED = false;
inline const std::string G_LOGGING_AREA = "hipdnn";
#endif

inline std::string generate_log_file_name(const std::string& logging_area)
{
    std::ostringstream oss;
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);
    oss << logging_area << "_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".log";
    return oss.str();
}

inline void setup_log_pattern(const std::string& logging_area)
{
    spdlog::set_pattern("[" + logging_area + "] [%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] %v");
}

inline std::string initialize_logger_with_output_file(const std::string& logging_area_name,
                                                      const std::string& log_file_directory)
{
    auto log_file_name = generate_log_file_name(logging_area_name);
    if(!log_file_directory.empty())
    {
        log_file_name = log_file_directory + "/" + log_file_name;
    }

    std::string logger_name = logging_area_name + "_logger";
    auto file_logger = spdlog::basic_logger_mt<spdlog::async_factory>(logger_name, log_file_name);
    file_logger->set_level(spdlog::level::off);

    spdlog::set_default_logger(file_logger);
    setup_log_pattern(logging_area_name);

    return log_file_name;
}

inline void initialize_logger_to_std_out(const std::string& logging_area_name)
{
    std::string logger_name = logging_area_name + "_logger";
    auto console_logger = spdlog::stdout_color_mt(logger_name);
    spdlog::set_default_logger(console_logger);

    setup_log_pattern(logging_area_name);
}

inline void set_log_level(const std::string& level)
{
    if(level == "off")
    {
        spdlog::set_level(spdlog::level::off);
    }
    else if(level == "info")
    {
        spdlog::set_level(spdlog::level::info);
    }
    else if(level == "warn")
    {
        spdlog::set_level(spdlog::level::warn);
    }
    else if(level == "error")
    {
        spdlog::set_level(spdlog::level::err);
    }
}

inline std::string
    initialize_logging_based_on_environment_variables(const std::string& logging_area)
{
    //values from getenv are a pointer to the environment table entry
    //We do not need to free them and modifying them results in undefined behavior
    const char* log_level = std::getenv("HIPDNN_LOG_LEVEL");
    const char* log_file_directory = std::getenv("HIPDNN_LOG_DIR");

    std::string output_file;
    if(log_file_directory != nullptr)
    {
        output_file
            = hipdnn::logging::initialize_logger_with_output_file(logging_area, log_file_directory);
    }
    else
    {
        hipdnn::logging::initialize_logger_to_std_out(logging_area);
    }

    if(log_level != nullptr)
    {
        hipdnn::logging::set_log_level(log_level);
    }
    else
    {
        hipdnn::logging::set_log_level("off");
    }

#ifdef ENABLE_BACKEND_LOGGING
    G_LOGGING_INITIALIZED = true;
#endif

    return output_file;
}

}