// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "logging.hpp"

#include <hipdnn_sdk/logging/component_formatter.hpp>
#include <hipdnn_sdk/logging/formatting.hpp>

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

// Anonymous namespace to encapsulate all state, keeping it private to this file.
std::mutex s_logging_init_mutex;
bool s_logging_initialized = false;
const std::string S_BACKEND_LOGGER_NAME = "hipdnn_backend";
const std::string S_CALLBACK_RECEIVER_LOGGER_NAME = "hipdnn_callback_receiver";

} // namespace

void initialize()
{
    // The lock guard ensures that even if multiple threads call the callback
    // simultaneously, the initialization logic only runs once.
    std::lock_guard<std::mutex> lock(s_logging_init_mutex);
    if(s_logging_initialized)
    {
        return;
    }

    const char* log_level = std::getenv("HIPDNN_LOG_LEVEL");
    const char* log_file_path = std::getenv("HIPDNN_LOG_FILE");

    if(log_level != nullptr && std::string(log_level) == "off")
    {
        s_logging_initialized = true; // Mark as initialized to prevent re-entry
        return;
    }

    spdlog::init_thread_pool(8192, 1);

    std::shared_ptr<spdlog::sinks::sink> shared_sink;
    if(log_file_path != nullptr && !std::string(log_file_path).empty())
    {
        shared_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file_path, false);
    }
    else
    {
        shared_sink = std::make_shared<spdlog::sinks::stderr_color_sink_mt>();
    }

    auto formatter = std::make_unique<hipdnn::logging::Component_formatter>();

    auto backend_logger = std::make_shared<spdlog::async_logger>(
        S_BACKEND_LOGGER_NAME, shared_sink, spdlog::thread_pool());
    backend_logger->set_formatter(formatter->clone());
    spdlog::register_logger(backend_logger);

    auto callback_receiver_logger = std::make_shared<spdlog::async_logger>(
        S_CALLBACK_RECEIVER_LOGGER_NAME, shared_sink, spdlog::thread_pool());
    callback_receiver_logger->set_formatter(std::move(formatter));
    spdlog::register_logger(callback_receiver_logger);

    if(log_level != nullptr)
    {
        if(std::string(log_level) == "info")
        {
            spdlog::set_level(spdlog::level::info);
        }
        else if(std::string(log_level) == "warn")
        {
            spdlog::set_level(spdlog::level::warn);
        }
        else if(std::string(log_level) == "error")
        {
            spdlog::set_level(spdlog::level::err);
        }
    }
    else
    {
        spdlog::set_level(spdlog::level::off);
    }
    s_logging_initialized = true;
}

std::shared_ptr<spdlog::logger> get_callback_receiver_logger()
{
    // This function simply retrieves the logger. It assumes initialize() has been called.
    return spdlog::get(S_CALLBACK_RECEIVER_LOGGER_NAME);
}

std::shared_ptr<spdlog::logger> get_logger()
{
    // This function simply retrieves the logger. It assumes initialize() has been called.
    return spdlog::get(S_BACKEND_LOGGER_NAME);
}

void cleanup()
{
    std::lock_guard<std::mutex> lock(s_logging_init_mutex);
    spdlog::shutdown(); // Flushes, stops thread pool, and drops all loggers.
    s_logging_initialized = false;
}

} // namespace logging
} // namespace hipdnn_backend