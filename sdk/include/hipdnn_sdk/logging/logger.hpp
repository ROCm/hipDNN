// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "callback_sink.hpp"
#include "callback_types.h"
#include "formatting.hpp"
#include <algorithm>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <regex>


#ifdef ENABLE_BACKEND_LOGGING

#define HIPDNN_LOG_INFO(...)                                                                      \
    if(!hipdnn::logging::g_logging_initialized)                                                   \
    {                                                                                             \
        hipdnn::logging::initialize_logging_based_on_environment_variables(                       \
            hipdnn::logging::G_LOGGING_AREA);                                                     \
    }                                                                                             \
    if(hipdnn::logging::g_backend_logger)                                                         \
    {                                                                                             \
        hipdnn::logging::g_backend_logger->info(__VA_ARGS__);                                     \
    }

#define HIPDNN_LOG_WARN(...)                                                                      \
    if(!hipdnn::logging::g_logging_initialized)                                                   \
    {                                                                                             \
        hipdnn::logging::initialize_logging_based_on_environment_variables(                       \
            hipdnn::logging::G_LOGGING_AREA);                                                     \
    }                                                                                             \
    if(hipdnn::logging::g_backend_logger)                                                         \
    {                                                                                             \
        hipdnn::logging::g_backend_logger->warn(__VA_ARGS__);                                     \
    }

#define HIPDNN_LOG_ERROR(...)                                                                     \
    if(!hipdnn::logging::g_logging_initialized)                                                   \
    {                                                                                             \
        hipdnn::logging::initialize_logging_based_on_environment_variables(                       \
            hipdnn::logging::G_LOGGING_AREA);                                                     \
    }                                                                                             \
    if(hipdnn::logging::g_backend_logger)                                                         \
    {                                                                                             \
        hipdnn::logging::g_backend_logger->error(__VA_ARGS__);                                    \
    }

#define HIPDNN_LOG_INFO_WITH_HANDLE(handle, ...)                                                  \
    if(handle)                                                                                    \
    {                                                                                             \
        throw not_implemented_exception("handle logging not implemented yet.");                   \
    }
#else
// No-op if COMPONENT_NAME is not defined or if the logger is not initialized.
#ifndef COMPONENT_NAME
    #define _HIPDNN_INTERNAL_LOG_ACTION(level, ...) do { } while(0)
#else
    #define _HIPDNN_INTERNAL_LOG_ACTION(level, ...) \
        do { \
            if (auto logger = spdlog::get(COMPONENT_NAME)) { \
                logger->level(__VA_ARGS__); \
            } \
        } while(0)
#endif

#define HIPDNN_LOG_INFO(...) _HIPDNN_INTERNAL_LOG_ACTION(info, __VA_ARGS__)
#define HIPDNN_LOG_WARN(...) _HIPDNN_INTERNAL_LOG_ACTION(warn, __VA_ARGS__)
#define HIPDNN_LOG_ERROR(...) _HIPDNN_INTERNAL_LOG_ACTION(error, __VA_ARGS__)
#define HIPDNN_LOG_INFO_WITH_HANDLE(handle, ...)
#endif

namespace hipdnn::logging
{
#ifdef ENABLE_BACKEND_LOGGING
inline bool g_logging_initialized = false;
inline std::string output_file;
inline std::mutex g_logging_init_mutex;
inline const std::string G_LOGGING_AREA = "hipdnn_backend";
inline std::shared_ptr<spdlog::logger> g_backend_logger;
inline std::shared_ptr<spdlog::logger> g_callback_receiver_logger;
#endif

inline std::string generate_rotated_log_file_name(const std::filesystem::path& base_path)
{
    // timestamp string in YYYYMMDD_HHMMSS format
    std::ostringstream oss;
    auto t = std::time(nullptr);
    std::tm tm_buf;
    auto tm = *localtime_r(&t, &tm_buf);
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S");
    const std::string timestamp = oss.str();

    std::filesystem::path new_path = base_path.parent_path();
    new_path /= base_path.stem().string() + "_" + timestamp + base_path.extension().string();

    return new_path.string();
}

#ifdef ENABLE_BACKEND_LOGGING

inline void rotate_log_files(const std::filesystem::path& base_path, size_t max_log_files)
{
    std::filesystem::path dir = base_path.parent_path();

    if(!dir.empty() && !std::filesystem::exists(dir))
    {
        std::filesystem::create_directories(dir);
    }

    if(std::filesystem::exists(dir))
    {
        std::vector<std::filesystem::path> existing_logs;
        const std::string stem_str = base_path.stem().string();
        const std::string ext_str_escaped
            = std::regex_replace(base_path.extension().string(), std::regex{"\\."}, "\\.");
        const std::regex pattern{stem_str + "_\\d{8}_\\d{6}" + ext_str_escaped};

        for(const auto& entry : std::filesystem::directory_iterator(dir))
        {
            if(entry.is_regular_file()
               && std::regex_match(entry.path().filename().string(), pattern))
            {
                existing_logs.push_back(entry.path());
            }
        }

        std::ranges::sort(existing_logs);

        // Remove the oldest files until we are under the limit
        while(existing_logs.size() >= max_log_files)
        {
            std::filesystem::remove(existing_logs.front());
            existing_logs.erase(existing_logs.begin());
        }
    }
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

    const char* log_level = std::getenv("HIPDNN_LOG_LEVEL");
    const char* log_file_path = std::getenv("HIPDNN_LOG_FILE");

    if(g_logging_initialized || (log_level != nullptr && std::string(log_level) == "off"))
    {
        return output_file;
    }

    try
    {
        if(!spdlog::thread_pool())
        {
            spdlog::init_thread_pool(8192, 1);
        }
        
        // Define the separate sinks for the callback receiver and backend logger because they need different patterns.
        std::shared_ptr<spdlog::sinks::sink> sink_for_callback_receiver;
        std::shared_ptr<spdlog::sinks::sink> sink_for_backend_logger;

        if(log_file_path != nullptr && !std::string(log_file_path).empty())
        {
            const size_t max_log_files = 5;
            rotate_log_files(log_file_path, max_log_files);

            output_file = generate_rotated_log_file_name(log_file_path);

            sink_for_callback_receiver
                = std::make_shared<spdlog::sinks::basic_file_sink_mt>(output_file, false);
            sink_for_backend_logger
                = std::make_shared<spdlog::sinks::basic_file_sink_mt>(output_file, false);
        }
        else
        {
            output_file.clear();
            sink_for_callback_receiver = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
            sink_for_backend_logger = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        }

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
    catch(const std::filesystem::filesystem_error& ex)
    {
        cleanup_logging();
        return "";
    }
}

#endif

inline void initialize_callback_logging(const std::string& logging_area,
                                        hipdnnCallback_t callback_function)
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

    auto callback_logger
        = hipdnn::logging::create_async_callback_logger_mt(callback_function, logging_area);
    spdlog::register_logger(callback_logger);
}

}