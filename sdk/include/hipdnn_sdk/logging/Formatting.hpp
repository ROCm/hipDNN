// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace hipdnn_sdk::logging
{

/**
 * @brief Generates a standard log pattern string with component name.
 * @param[in] componentName Name of the component to include in the pattern.
 * @return Formatted pattern string for use with spdlog.
 */
inline std::string generatePatternString(const std::string& componentName)
{
    return "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] [" + componentName + "] %v";
}

} // namespace hipdnn_sdk::logging
