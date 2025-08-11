// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "shared_library.hpp"
#include "hipdnn_exception.hpp"
#include "logging/logging.hpp"
#include "platform_utils.hpp"

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

    // Check file extension and add prefix/suffix if needed
    if(modified_library_path.has_extension())
    {
        if(modified_library_path.extension() != hipdnn_sdk::utilities::SHARED_LIB_EXT)
        {
            throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                                   std::string("Invalid file extension. Expected ")
                                       + hipdnn_sdk::utilities::SHARED_LIB_EXT);
        }
    }
    else
    {
        auto library_name
            = hipdnn_sdk::utilities::get_library_name(modified_library_path.filename().string().c_str());
        modified_library_path = modified_library_path.parent_path() / library_name;
    }

    _library_path = std::filesystem::weakly_canonical(modified_library_path);

    // Needs to be a resolved, weakly canonical path at this point
    if(!std::filesystem::exists(_library_path))
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Shared libary: plugin file does not exist: "
                                   + _library_path.string());
    }

    HIPDNN_LOG_INFO(
        "Shared_library: Attempting to load shared library from final absolute path: {}",
        _library_path.string());

    _library_handle = platform_utils::open_library(_library_path);
}

void Shared_library::unload() noexcept
{
    if(_library_handle != nullptr)
    {
        platform_utils::close_library(_library_handle);
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

    return platform_utils::get_symbol(_library_handle, symbol_name.data());
}

const std::filesystem::path& Shared_library::library_path() const
{
    return _library_path;
}

} // namespace plugin
} // hipdnn_backend
