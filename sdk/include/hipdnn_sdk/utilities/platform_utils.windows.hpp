// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#ifdef _WIN32

#include <windows.h>

namespace hipdnn_sdk
{
namespace utilities
{

constexpr const char* SHARED_LIB_EXT = ".dll";
constexpr const char* LIB_PREFIX = "";
constexpr const char* EXECUTABLE_EXT = ".exe";

inline std::string get_env(const char* var, const char* default_value = nullptr)
{
    std::string result = default_value != nullptr ? default_value : "";

    GetEnvironmentVariableA(var, nullptr, 0);

    DWORD size = GetEnvironmentVariableA(var, nullptr, 0);
    if(size > 0)
    {
        char* dst = new char[size];
        GetEnvironmentVariableA(var, dst, size);
        result = dst;
        delete[] dst;
    }

    return result;
}

inline void set_env(const char* var, const char* value)
{
    if(value != nullptr)
    {
        SetEnvironmentVariableA(var, value);
    }
}

inline void unset_env(const char* var)
{
    SetEnvironmentVariableA(var, nullptr);
}

}
}

#else

#error "Do not include platform_utils.windows.hpp in non-windows builds"

#endif
