// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

namespace hipdnn_sdk
{
namespace utilities
{

#ifdef _WIN32

typedef HMODULE Plugin_lib_handle;
constexpr const char* SHARED_LIB_EXT = ".dll";
constexpr const char* LIB_PREFIX = "";

#elif defined(__linux__)

typedef void* Plugin_lib_handle;
constexpr const char* SHARED_LIB_EXT = ".so";
constexpr const char* LIB_PREFIX = "lib";

#else

#error "Unsupported platform"

#endif

inline std::string get_library_name(const char* library_base_name)
{
    return std::string(LIB_PREFIX) + library_base_name + SHARED_LIB_EXT;
}

}
}