// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "platform_utils.hpp"

#ifdef _WIN32

#include "hipdnn_exception.hpp"

namespace hipdnn_backend::platform_utils
{

std::filesystem::path get_current_module_directory()
{
    std::filesystem::path module_path;

    HMODULE module_handle = nullptr;
    if(GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                              | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCSTR>(&get_current_module_directory),
                          &module_handle)
       == TRUE)
    {
        char* dst = new char[MAX_PATH];
        DWORD len = GetModuleFileNameA(module_handle, dst, MAX_PATH);
        std::string module_path_str(dst);
        delete[] dst;

        if(len > 0 && len < MAX_PATH)
        {
            module_path = std::filesystem::path(module_path_str).parent_path();
        }
        else
        {
            throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module file name.");
        }
    }
    else
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR, "Failed to get module handle.");
    }

    return std::filesystem::weakly_canonical(std::filesystem::absolute(module_path));
}

Plugin_lib_handle open_library(const std::filesystem::path& library_path)
{
    Plugin_lib_handle handle = LoadLibraryW(library_path.wstring().c_str());
    if(handle == nullptr)
    {
        auto error_code = GetLastError();
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Failed to load library: " + library_path.string()
                                   + " (Error Code: " + std::to_string(error_code) + ")");
    }

    return handle;
}

void close_library(Plugin_lib_handle handle)
{
    FreeLibrary(handle);
}

void* get_symbol(Plugin_lib_handle handle, const char* symbol_name)
{
    void* symbol = reinterpret_cast<void*>(GetProcAddress(handle, symbol_name));
    if(symbol == nullptr)
    {
        auto error_code = GetLastError();
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get symbol: " + std::string(symbol_name)
                                   + " (Error Code: " + std::to_string(error_code) + ")");
    }

    return symbol;
}

}

#endif // _WIN32
