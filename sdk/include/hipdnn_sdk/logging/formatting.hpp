// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <string>

namespace hipdnn::logging
{

/**
 * @brief Generates a standard log pattern string with component name.
 * @param[in] component_name Name of the component to include in the pattern.
 * @return Formatted pattern string for use with spdlog.
 */
inline std::string generate_pattern_string(const std::string& component_name)
{
    return "[%Y-%m-%d %H:%M:%S.%e] [tid %t] [%l] [" + component_name + "] %v";
}

} // namespace hipdnn::logging
