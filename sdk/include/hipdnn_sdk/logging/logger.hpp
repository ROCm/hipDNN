// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "callback_sink.hpp"
#include "callback_types.h"
#include "formatting.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <ctime>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>

#ifdef ENABLE_BACKEND_LOGGING

#define HIPDNN_LOG_INFO(...)                                                \
    if(!hipdnn::logging::g_logging_initialized)                             \
    {                                                                       \
        hipdnn::logging::initialize_logging_based_on_environment_variables( \
            hipdnn::logging::G_LOGGING_AREA);                               \
    }                                                                       \
    hipdnn::logging::g_backend_logger->info(__VA_ARGS__);

#define HIPDNN_LOG_WARN(...)                                                \
    if(!hipdnn::logging::g_logging_initialized)                             \
    {                                                                       \
        hipdnn::logging::initialize_logging_based_on_environment_variables( \
            hipdnn::logging::G_LOGGING_AREA);                               \
    }                                                                       \
    hipdnn::logging::g_backend_logger->warn(__VA_ARGS__);

#define HIPDNN_LOG_ERROR(...)                                               \
    if(!hipdnn::logging::g_logging_initialized)                             \
    {                                                                       \
        hipdnn::logging::initialize_logging_based_on_environment_variables( \
            hipdnn::logging::G_LOGGING_AREA);                               \
    }                                                                       \
    hipdnn::logging::g_backend_logger->error(__VA_ARGS__);

#define HIPDNN_LOG_INFO_WITH_HANDLE(handle, ...)                                \
    if(handle)                                                                  \
    {                                                                           \
        throw not_implemented_exception("handle logging not implemented yet."); \
    }
#else
#define HIPDNN_LOG_INFO(...) spdlog::default_logger_raw()->info(__VA_ARGS__);
#define HIPDNN_LOG_WARN(...) spdlog::default_logger_raw()->warn(__VA_ARGS__);
#define HIPDNN_LOG_ERROR(...) spdlog::default_logger_raw()->error(__VA_ARGS__);
#define HIPDNN_LOG_INFO_WITH_HANDLE(handle, ...)
#endif

namespace hipdnn::logging
{
#ifdef ENABLE_BACKEND_LOGGING
inline bool g_logging_initialized = false; // the compiler wants lowercase
inline std::string output_file;
inline std::mutex g_logging_init_mutex;
inline const std::string G_LOGGING_AREA = "hipdnn_backend";
inline std::shared_ptr<spdlog::logger> g_backend_logger;
inline std::shared_ptr<spdlog::logger> g_callback_receiver_logger;
#endif

inline std::string generate_log_file_name()
{
    std::ostringstream oss;
    auto t = std::time(nullptr);
    std::tm tm_buf;
    auto tm = *localtime_r(&t, &tm_buf);
    oss << "hipdnn_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".log";
    return oss.str();
}

#ifdef ENABLE_BACKEND_LOGGING

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

inline void cleanup_logging()
{
    if(g_backend_logger)
    {
        spdlog::drop(g_backend_logger->name());
        g_backend_logger.reset();
    }

    if(g_callback_receiver_logger)
    {
        spdlog::drop(g_callback_receiver_logger->name());
        g_callback_receiver_logger.reset();
    }

    output_file.clear();
    g_logging_initialized = false;
}

inline std::string
    initialize_logging_based_on_environment_variables(const std::string& component_name)
{
    std::lock_guard<std::mutex> lock(g_logging_init_mutex);

    if(g_logging_initialized)
    {
        return output_file;
    }

    const char* log_level = std::getenv("HIPDNN_LOG_LEVEL");
    const char* log_file_path = std::getenv("HIPDNN_LOG_FILE");

    if(log_file_path != nullptr && !std::string(log_file_path).empty())
    {
        output_file = log_file_path;
    }
    else
    {
        output_file = generate_log_file_name();
    }

    try
    {
        if(!spdlog::thread_pool())
        {
            spdlog::init_thread_pool(8192, 1);
        }

        std::shared_ptr<spdlog::sinks::sink> sink_for_callback_receiver;
        std::shared_ptr<spdlog::sinks::sink> sink_for_backend_logger;

        sink_for_callback_receiver
            = std::make_shared<spdlog::sinks::basic_file_sink_mt>(output_file);
        sink_for_backend_logger = std::make_shared<spdlog::sinks::basic_file_sink_mt>(output_file);

        g_backend_logger = std::make_shared<spdlog::async_logger>(
            component_name, sink_for_backend_logger, spdlog::thread_pool());
        g_backend_logger->set_pattern(generate_pattern_string(component_name));
        ::spdlog::register_logger(g_backend_logger);
        g_backend_logger->flush_on(spdlog::level::info);

        g_callback_receiver_logger = std::make_shared<spdlog::async_logger>(
            "hipdnn", sink_for_callback_receiver, spdlog::thread_pool());
        g_callback_receiver_logger->set_pattern("%v");
        ::spdlog::register_logger(g_callback_receiver_logger);
        g_callback_receiver_logger->flush_on(spdlog::level::info);

        if(log_level != nullptr)
        {
            hipdnn::logging::set_log_level(log_level);
        }
        else
        {
            hipdnn::logging::set_log_level("off");
        }

        g_logging_initialized = true;

        return output_file;
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        cleanup_logging();
        return "";
    }
}

#endif

inline void initialize_callback_logging(const std::string& logging_area,
                                        hipdnnCallback_t callback_function,
                                        void* user_data = nullptr)
{
    static std::mutex callback_init_mutex;
    std::lock_guard<std::mutex> lock(callback_init_mutex);

    if(spdlog::get(logging_area))
    {
        spdlog::drop(logging_area);
    }

    if(!spdlog::thread_pool())
    {
        spdlog::init_thread_pool(8192, 1);
    }

    auto callback_sink = hipdnn::logging::create_callback_logger_mt(
        callback_function, user_data, logging_area);

#ifndef ENABLE_BACKEND_LOGGING
    spdlog::set_default_logger(callback_sink);
#endif
}

}
