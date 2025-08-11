// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <string>
#include <hipdnn_sdk/utilities/platform_path_utils.hpp>

namespace hipdnn_backend::platform_utils
{

#ifdef _WIN32

typedef HMODULE Plugin_lib_handle;

#elif defined(__linux__)

typedef void* Plugin_lib_handle;

#else

#error "Unsupported platform"

#endif

std::filesystem::path get_current_module_directory();

Plugin_lib_handle open_library(const std::filesystem::path& library_path);
void close_library(Plugin_lib_handle handle);
void* get_symbol(Plugin_lib_handle handle, const char* symbol_name);

}
