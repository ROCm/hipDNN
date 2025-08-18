// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(__linux__)
namespace hipdnn_sdk
{
namespace utilities
{

constexpr const char* SHARED_LIB_EXT = ".so";
constexpr const char* LIB_PREFIX = "lib";
constexpr const char* EXECUTABLE_EXT = "";

inline std::string get_env(const char* var, const char* default_value = nullptr)
{
    std::string result = default_value != nullptr ? default_value : "";

    const char* value = std::getenv(var);

    if(value != nullptr)
    {
        result = value;
    }

    return result;
}

inline void set_env(const char* var, const char* value)
{
    if(value != nullptr)
    {
        setenv(var, value, 1);
    }
}

inline void unset_env(const char* var)
{
    unsetenv(var);
}

}
}

#else

#error "Do not include platform_utils.linux.hpp in non-linux builds"

#endif
