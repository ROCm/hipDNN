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

inline std::string getLibraryName(const char* libraryBaseName)
{
    return std::string(LIB_PREFIX) + libraryBaseName + SHARED_LIB_EXT;
}

inline std::string getExecutableName(const char* executableBaseName)
{
    return std::string(executableBaseName) + EXECUTABLE_EXT;
}

}
}
