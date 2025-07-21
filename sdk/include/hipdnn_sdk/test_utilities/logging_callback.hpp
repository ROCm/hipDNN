// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_sdk/logging/callback_types.h>

#include <cstdlib>
#include <iostream>
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
    const char* log_level_env = std::getenv("HIPDNN_LOG_LEVEL");
    std::string log_level_str = log_level_env ? log_level_env : "off";

    if(log_level_str == "off")
    {
        return;
    }

    hipdnnSeverity_t configured_level = string_to_severity(log_level_str);

    if(severity >= configured_level)
    {
        std::cerr << message << std::endl;
    }
}

} // namespace logging_test_utils