// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifndef _WIN32
#include <dlfcn.h>
#endif
#include <iostream>
#include <stdexcept>

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
    if(!load(library_path))
    {
        // TODO we do not have an exception class yet, so we just throw std::runtime_error
        throw std::runtime_error("Failed to load shared library: " + library_path.string());
    }
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
        _library_handle       = other._library_handle;
        other._library_handle = nullptr;
    }
    return *this;
}

// This function loads a shared library from the specified path.
// On Windows, it adds a ".dll" extension if no extension exists.
// On Linux, it adds a "lib" prefix to the filename and a ".so" extension if no extension exists.
bool Shared_library::load(const std::filesystem::path& library_path)
{
    if(_library_handle != nullptr)
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: Library is already loaded." << '\n';
        return false;
    }

    auto modified_library_path = library_path;

    // Check file extension
    if(modified_library_path.has_extension())
    {
#ifdef _WIN32
        if(modified_library_path.extension() != ".dll")
        {
            // TODO We do not have a logger yet, so we just print to stderr
            std::cerr << "Error: Invalid file extension. Expected '.dll'." << '\n';
            return false;
        }
#elif defined(__linux__)
        if(modified_library_path.extension() != ".so")
        {
            // TODO We do not have a logger yet, so we just print to stderr
            std::cerr << "Error: Invalid file extension. Expected '.so'." << '\n';
            return false;
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
        auto filename         = std::filesystem::path("lib") += modified_library_path.filename();
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
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: Failed to load library: " << modified_library_path.string()
                  << " (Error Code: " << errorCode << ")" << '\n';
        return false;
    }
#elif defined(__linux__)
    _library_handle = dlopen(modified_library_path.string().c_str(), RTLD_LAZY);
    if(_library_handle == nullptr)
    {
        const char* error = dlerror();
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: Failed to load library: " << modified_library_path.string()
                  << " (Error: " << (error != nullptr ? error : "Unknown error") << ")" << '\n';
        return false;
    }
#else
#error "Unsupported platform"
#endif

    return true;
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
        // TODO we do not have an exception class yet, so we just throw std::runtime_error
        throw std::runtime_error("Library is not loaded. Cannot get symbol: "
                                 + std::string(symbol_name));
    }

#ifdef _WIN32
    void* symbol = GetProcAddress(_library_handle, symbol_name.data());
    if(symbol == nullptr)
    {
        auto errorCode = GetLastError();
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Failed to get symbol: " << symbol_name << " (Error Code: " << errorCode << ")"
                  << '\n';
        return nullptr;
    }
#elif defined(__linux__)
    void* symbol = dlsym(_library_handle, symbol_name.data());
    if(symbol == nullptr)
    {
        const char* error = dlerror();
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Failed to get symbol: " << symbol_name << " - "
                  << (error != nullptr ? error : "Unknown error") << '\n';
        return nullptr;
    }
#else
#error "Unsupported platform"
#endif
    return symbol;
}

} // namespace plugin
} // hipdnn_backend
