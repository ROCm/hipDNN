// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace hipdnn_sdk
{
namespace utilities
{

#ifdef _WIN32

constexpr const char* SHARED_LIB_EXT = ".dll";
constexpr const char* LIB_PREFIX = "";

inline std::string get_env(const char* var, const char* default_value = nullptr)
{
    std::string result = default_value != nullptr ? default_value : "";

    size_t size = 0;
    char* dst = nullptr;
    getenv_s(&size, nullptr, 0, var);
    if (size > 0)
    {
        dst = new char[size];
        getenv_s(&size, dst, size, var);
        result = dst;
        delete[] dst;
    }
    
    return result;
}

#elif defined(__linux__)

constexpr const char* SHARED_LIB_EXT = ".so";
constexpr const char* LIB_PREFIX = "lib";

inline std::string get_env(const char* var, const char* default_value = nullptr)
{
    std::string result = default_value != nullptr ? default_value : "";

    const char* value = std::getenv(var);

    if (value != nullptr)
    {
        result = value;
    }

    return result;
}

#else

#error "Unsupported platform"

#endif

inline std::string get_library_name(const char* library_base_name)
{
    return std::string(LIB_PREFIX) + library_base_name + SHARED_LIB_EXT;
}

}
}