// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <string>

namespace hipdnn_backend::platform_utils
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

std::filesystem::path get_current_module_directory();

Plugin_lib_handle open_library(const std::filesystem::path& library_path);
void close_library(Plugin_lib_handle handle);
void* get_symbol(Plugin_lib_handle handle, const char* symbol_name);

}
