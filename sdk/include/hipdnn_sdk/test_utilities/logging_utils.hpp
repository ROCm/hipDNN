// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/logging/callback_types.h>
#include <hipdnn_sdk/logging/component_formatter.hpp>

#include <cstdlib>
#include <iostream>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <string>

namespace logging_test_utils
{

inline hipdnnSeverity_t string_to_severity(const std::string& level_str)
{
    if(level_str == "info")
    {
        return HIPDNN_SEV_INFO;
    }
    if(level_str == "warn")
    {
        return HIPDNN_SEV_WARN;
    }
    if(level_str == "error")
    {
        return HIPDNN_SEV_ERROR;
    }
    if(level_str == "fatal")
    {
        return HIPDNN_SEV_FATAL;
    }
    return HIPDNN_SEV_OFF;
}

inline void test_logging_callback(hipdnnSeverity_t severity, const char* message)
{
#ifndef DISABLE_TEST_LOGGING

    const char* log_level_env = std::getenv("HIPDNN_LOG_LEVEL");
    std::string log_level_str = log_level_env != nullptr ? log_level_env : "off";

    if(log_level_str == "off")
    {
        return;
    }

    hipdnnSeverity_t configured_level = string_to_severity(log_level_str);

    if(severity >= configured_level)
    {
        std::cerr << message << '\n';
    }
#endif
}

inline void initialize_spdlog_default_logger(const std::string& component_name)
{
#ifndef DISABLE_TEST_LOGGING
    spdlog::drop_all();
    auto logger = spdlog::stdout_color_mt(component_name);
    logger->set_formatter(std::make_unique<hipdnn::logging::Component_formatter>());
    spdlog::set_default_logger(logger);
    spdlog::set_level(spdlog::level::info); // Set default log level
#endif
}

} // namespace logging_test_utils