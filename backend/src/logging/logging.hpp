// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <spdlog/spdlog.h>

#ifdef HIPDNN_BACKEND_COMPILATION
#define HIPDNN_LOG_INFO(...)                                 \
    hipdnn_backend::logging::initialize();                   \
    if(auto _logger = hipdnn_backend::logging::get_logger()) \
    {                                                        \
        _logger->info(__VA_ARGS__);                          \
    }

#define HIPDNN_LOG_WARN(...)                                 \
    hipdnn_backend::logging::initialize();                   \
    if(auto _logger = hipdnn_backend::logging::get_logger()) \
    {                                                        \
        _logger->warn(__VA_ARGS__);                          \
    }

#define HIPDNN_LOG_ERROR(...)                                \
    hipdnn_backend::logging::initialize();                   \
    if(auto _logger = hipdnn_backend::logging::get_logger()) \
    {                                                        \
        _logger->error(__VA_ARGS__);                         \
    }
#endif

namespace hipdnn_backend
{
namespace logging
{

void initialize();

std::shared_ptr<spdlog::logger> get_logger();

std::shared_ptr<spdlog::logger> get_callback_receiver_logger();

void cleanup();

} // namespace logging
} // namespace hipdnn_backend