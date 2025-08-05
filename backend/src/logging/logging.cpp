// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "logging.hpp"

#include <hipdnn_sdk/logging/component_formatter.hpp>
#include <hipdnn_sdk/logging/formatting.hpp>
#include <iostream>

#include <hipdnn_sdk/logging/callback_types.h>
#include <spdlog/async.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <mutex>

namespace hipdnn_backend
{
namespace logging
{
namespace
{

// Could refactor this to a class with a single static instance.
// The benefit would be a destructor to cleanup logging.
std::mutex s_logging_init_mutex;
bool s_logging_initialized = false;
const std::string S_BACKEND_LOGGER_NAME = "hipdnn_backend";
const std::string S_CALLBACK_RECEIVER_LOGGER_NAME = "hipdnn_callback_receiver";

} // namespace

void initialize()
{
    try
    {
        std::lock_guard<std::mutex> lock(s_logging_init_mutex);
        if(s_logging_initialized)
        {
            return;
        }

        const char* log_level = std::getenv("HIPDNN_LOG_LEVEL");
        const char* log_file_path = std::getenv("HIPDNN_LOG_FILE");

        // It doesn't need to return if log_level == off, but it avoids unnecessary initialization
        if(log_level == nullptr || std::string(log_level) == "off")
        {
            s_logging_initialized = true;
            return;
        }

        if(!spdlog::thread_pool())
        {
            spdlog::init_thread_pool(8192, 1);
        }

        std::shared_ptr<spdlog::sinks::sink> shared_sink;
        if(log_file_path != nullptr && !std::string(log_file_path).empty())
        {
            shared_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, false);
        }
        else
        {
            shared_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
        }

        auto backend_logger = std::make_shared<spdlog::async_logger>(
            S_BACKEND_LOGGER_NAME, shared_sink, spdlog::thread_pool());

        // In spdlog, the formatting is a property of the underlying sink, not the logger.
        // However, we need one destination sink for thread safety because the mutex is attached to the sink.
        // Therefore, we implement a custom formatter to have distinct formatting for the backend, which does not use a callback sink.
        backend_logger->set_formatter(std::make_unique<hipdnn::logging::Component_formatter>());
        spdlog::register_logger(backend_logger);

        auto callback_receiver_logger = std::make_shared<spdlog::async_logger>(
            S_CALLBACK_RECEIVER_LOGGER_NAME, shared_sink, spdlog::thread_pool());
        spdlog::register_logger(callback_receiver_logger);

        set_log_level(log_level);

        s_logging_initialized = true;

        return;
    }
    catch(const spdlog::spdlog_ex& ex)
    {
        cleanup();
        std::cerr << "Logging initialization failed: " << ex.what() << "\n";
    }
}

void cleanup()
{
    std::lock_guard<std::mutex> lock(s_logging_init_mutex);
    spdlog::shutdown();
    s_logging_initialized = false;
}

void set_log_level(const std::string& level)
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
    else if(level == "fatal")
    {
        spdlog::set_level(spdlog::level::critical);
    }
}

std::shared_ptr<spdlog::logger> get_callback_receiver_logger()
{
    return spdlog::get(S_CALLBACK_RECEIVER_LOGGER_NAME);
}

std::shared_ptr<spdlog::logger> get_backend_logger()
{
    return spdlog::get(S_BACKEND_LOGGER_NAME);
}

void hipdnn_logging_callback(hipdnnSeverity_t severity, const char* msg)
{
    initialize();

    if(auto logger = get_callback_receiver_logger())
    {
        switch(severity)
        {
        case HIPDNN_SEV_FATAL:
            logger->critical(msg);
            break;
        case HIPDNN_SEV_ERROR:
            logger->error(msg);
            break;
        case HIPDNN_SEV_WARN:
            logger->warn(msg);
            break;
        case HIPDNN_SEV_OFF:
            break;
        default:
            logger->info(msg);
            break;
        }
    }
}

} // namespace logging
} // namespace hipdnn_backend
