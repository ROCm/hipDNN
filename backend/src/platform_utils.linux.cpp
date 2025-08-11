// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "platform_utils.hpp"

#if defined(__linux__)

#include "hipdnn_exception.hpp"
#include <dlfcn.h>

namespace hipdnn_backend::platform_utils
{

std::filesystem::path get_current_module_directory()
{
    std::filesystem::path module_path;

    Dl_info info;
    if(dladdr(reinterpret_cast<void const*>(&get_current_module_directory), &info) != 0
       && info.dli_fname != nullptr && info.dli_fname[0] != '\0')
    {
        module_path = std::filesystem::path(info.dli_fname).parent_path();
    }
    else
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module file name.");
    }

    return std::filesystem::weakly_canonical(std::filesystem::absolute(module_path));
}

Plugin_lib_handle open_library(const std::filesystem::path& library_path)
{
#if __has_feature(address_sanitizer)
    // Address Sanitizer does not support RTLD_DEEPBIND, so we use RTLD_NOW only
    Plugin_lib_handle handle = dlopen(library_path.string().c_str(), RTLD_NOW);
#else
    Plugin_lib_handle handle = dlopen(library_path.string().c_str(), RTLD_NOW | RTLD_DEEPBIND);
#endif

    if(handle == nullptr)
    {
        const char* error = dlerror();
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Failed to load library: " + library_path.string() + " (Error: "
                                   + (error != nullptr ? std::string(error) : "Unknown error")
                                   + ")");
    }

    return handle;
}

void close_library(Plugin_lib_handle handle)
{
    dlclose(handle);
}

void* get_symbol(Plugin_lib_handle handle, const char* symbol_name)
{
    void* symbol = dlsym(handle, symbol_name);
    if(symbol == nullptr)
    {
        const char* error = dlerror();
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get symbol: " + std::string(symbol_name) + " (Error: "
                                   + (error != nullptr ? error : "Unknown error") + ")");
    }
    return symbol;
}

}

#endif // defined(__linux__)
