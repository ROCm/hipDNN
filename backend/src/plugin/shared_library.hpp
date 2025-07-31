// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <string_view>
#ifdef _WIN32
#include <windows.h>
#endif

namespace hipdnn_backend
{
namespace plugin
{

class Shared_library
{
public:
    static std::filesystem::path get_current_module_directory();

    Shared_library();
    explicit Shared_library(const std::filesystem::path& library_path);
    Shared_library(Shared_library&& other) noexcept;
    Shared_library(const Shared_library&) = delete;

    ~Shared_library();

    Shared_library& operator=(Shared_library&& other) noexcept;
    Shared_library& operator=(const Shared_library&) = delete;

    // This function loads a shared library from the specified path.
    // On Windows, it adds a ".dll" extension if no extension exists.
    // On Linux, it adds a "lib" prefix to the filename and a ".so" extension if no extension exists.
    void load(const std::filesystem::path& library_path);

    // This function unloads the shared library.
    void unload() noexcept;

    void* get_symbol(std::string_view symbol_name) const;

    template <class T>
    T get_symbol(std::string_view symbol_name) const
    {
        void* symbol = get_symbol(symbol_name);
        return reinterpret_cast<T>(symbol);
    }

    const std::filesystem::path& library_path() const;

private:
    std::filesystem::path _library_path;
#ifdef _WIN32
    HMODULE _library_handle;
#elif defined(__linux__)
    void* _library_handle;
#endif
};

} // namespace plugin
} // hipdnn_backend
