// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include "hipdnn_exception.hpp"
#include "shared_library.hpp"

namespace hipdnn_backend
{
namespace plugin
{

Shared_library::Shared_library()
    : _library_handle(nullptr)
{
    // Default constructor does nothing
}

Shared_library::Shared_library(const std::filesystem::path& library_path)
    : _library_handle(nullptr)
{
    load(library_path);
}

Shared_library::Shared_library(Shared_library&& other) noexcept
    : _library_handle(other._library_handle)
{
    other._library_handle = nullptr;
}

Shared_library::~Shared_library()
{
    unload();
}

Shared_library& Shared_library::operator=(Shared_library&& other) noexcept
{
    if(this != &other)
    {
        // Release current resources
        unload();

        // Transfer ownership
        _library_handle = other._library_handle;
        other._library_handle = nullptr;
    }
    return *this;
}

// This function loads a shared library from the specified path.
// On Windows, it adds a ".dll" extension if no extension exists.
// On Linux, it adds a "lib" prefix to the filename and a ".so" extension if no extension exists.
void Shared_library::load(const std::filesystem::path& library_path)
{
    if(_library_handle != nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR, "Library is already loaded.");
    }

    auto modified_library_path = library_path;

    // Check file extension
    if(modified_library_path.has_extension())
    {
#ifdef _WIN32
        if(modified_library_path.extension() != ".dll")
        {
            throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                                   "Invalid file extension. Expected '.dll'.");
        }
#elif defined(__linux__)
        if(modified_library_path.extension() != ".so")
        {
            throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                                   "Invalid file extension. Expected '.so'.");
        }
#else
#error "Unsupported platform"
#endif
    }
    else
    {
#ifdef _WIN32
        // Add ".dll" extension if no extension exists
        modified_library_path.replace_extension(".dll");
#elif defined(__linux__)
        // Add "lib" prefix to the filename and ".so" extension if no extension exists
        auto filename = std::filesystem::path("lib") += modified_library_path.filename();
        modified_library_path = modified_library_path.parent_path() / filename;
        modified_library_path.replace_extension(".so");
#else
#error "Unsupported platform"
#endif
    }

#ifdef _WIN32
    _library_handle = LoadLibraryW(modified_library_path.wstring().c_str());
    if(_library_handle == nullptr)
    {
        auto errorCode = GetLastError();
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to load library: " + modified_library_path.string()
                                   + " (Error Code: " + std::to_string(errorCode) + ")");
    }
#elif defined(__linux__)
    _library_handle = dlopen(modified_library_path.string().c_str(), RTLD_LAZY);
    if(_library_handle == nullptr)
    {
        const char* error = dlerror();
        throw Hipdnn_exception(
            HIPDNN_STATUS_PLUGIN_ERROR,
            "Failed to load library: " + modified_library_path.string()
                + " (Error: " + (error != nullptr ? std::string(error) : "Unknown error") + ")");
    }
#else
#error "Unsupported platform"
#endif
}

void Shared_library::unload() noexcept
{
    if(_library_handle != nullptr)
    {
#ifdef _WIN32
        FreeLibrary(_library_handle);
#elif defined(__linux__)
        dlclose(_library_handle);
#else
#error "Unsupported platform"
#endif
        _library_handle = nullptr;
    }
}

void* Shared_library::get_symbol(std::string_view symbol_name) const
{
    if(_library_handle == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Library is not loaded. Cannot get symbol: "
                                   + std::string(symbol_name));
    }

#ifdef _WIN32
    void* symbol = GetProcAddress(_library_handle, symbol_name.data());
    if(symbol == nullptr)
    {
        auto errorCode = GetLastError();
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get symbol: " + std::string(symbol_name)
                                   + " (Error Code: " + std::to_string(errorCode) + ")");
    }
#elif defined(__linux__)
    void* symbol = dlsym(_library_handle, symbol_name.data());
    if(symbol == nullptr)
    {
        const char* error = dlerror();
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get symbol: " + std::string(symbol_name) + " (Error: "
                                   + (error != nullptr ? error : "Unknown error") + ")");
    }
#else
#error "Unsupported platform"
#endif
    return symbol;
}

} // namespace plugin
} // hipdnn_backend
