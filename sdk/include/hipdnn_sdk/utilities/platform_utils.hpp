// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#ifdef _WIN32

#include "platform_utils.windows.hpp"

#elif defined(__linux__)

#include "platform_utils.linux.hpp"

#else

#error "Unsupported platform"

#endif

namespace hipdnn_sdk
{
namespace utilities
{

inline std::string get_library_name(const char* library_base_name)
{
    return std::string(LIB_PREFIX) + library_base_name + SHARED_LIB_EXT;
}

inline std::string get_executable_name(const char* executable_base_name)
{
    return std::string(executable_base_name) + EXECUTABLE_EXT;
}

}
}
