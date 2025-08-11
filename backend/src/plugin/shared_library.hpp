// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include "platform_utils.hpp"
#include <filesystem>
#include <string_view>

namespace hipdnn_backend
{
namespace plugin
{

class Shared_library
{
public:
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

    hipdnn_backend::platform_utils::Plugin_lib_handle _library_handle;
};

} // namespace plugin
} // hipdnn_backend
