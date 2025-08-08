// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <string>
#include <hipdnn_sdk/utilities/platform_path_utils.hpp>

namespace hipdnn_backend::platform_utils
{

std::filesystem::path get_current_module_directory();

hipdnn_sdk::utilities::Plugin_lib_handle open_library(const std::filesystem::path& library_path);
void close_library(hipdnn_sdk::utilities::Plugin_lib_handle handle);
void* get_symbol(hipdnn_sdk::utilities::Plugin_lib_handle handle, const char* symbol_name);

}
