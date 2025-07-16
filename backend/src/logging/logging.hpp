// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <memory>
#include <spdlog/spdlog.h>

#ifdef HIPDNN_BACKEND_COMPILATION
#define _HIPDNN_BACKEND_LOG_ACTION(level, ...)                   \
    do                                                           \
    {                                                            \
        hipdnn_backend::logging::initialize();                   \
        if(auto _logger = hipdnn_backend::logging::get_logger()) \
        {                                                        \
            _logger->level(__VA_ARGS__);                         \
        }                                                        \
    } while(0)

#define HIPDNN_LOG_INFO(...) _HIPDNN_BACKEND_LOG_ACTION(info, __VA_ARGS__)
#define HIPDNN_LOG_WARN(...) _HIPDNN_BACKEND_LOG_ACTION(warn, __VA_ARGS__)
#define HIPDNN_LOG_ERROR(...) _HIPDNN_BACKEND_LOG_ACTION(error, __VA_ARGS__)
#endif

namespace hipdnn_backend
{
namespace logging
{

void initialize();

// maybe just make a constructor and destructor
void cleanup();

std::shared_ptr<spdlog::logger> get_logger();

std::shared_ptr<spdlog::logger> get_callback_receiver_logger();

} // namespace logging
} // namespace hipdnn_backend