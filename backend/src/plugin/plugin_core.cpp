// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cassert>

#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{

Plugin_base::Plugin_base(Shared_library&& lib)
    : _lib(std::move(lib))
{
    resolve_symbols();
}

void Plugin_base::resolve_symbols()
{
    const auto func_name_get_name = "hipdnnPluginGetName";
    _func_get_name = _lib.get_symbol<decltype(_func_get_name)>(func_name_get_name);

    const auto func_name_get_version = "hipdnnPluginGetVersion";
    _func_get_version = _lib.get_symbol<decltype(_func_get_version)>(func_name_get_version);

    const auto func_name_get_type = "hipdnnPluginGetType";
    _func_get_type = _lib.get_symbol<decltype(_func_get_type)>(func_name_get_type);

    const auto func_name_get_last_error_str = "hipdnnPluginGetLastErrorString";
    _func_get_last_error_str
        = _lib.get_symbol<decltype(_func_get_last_error_str)>(func_name_get_last_error_str);

#ifndef NDEBUG
    _initialized = true;
#endif
}

std::string_view Plugin_base::name() const
{
    assert(_initialized);
    const char* name;
    auto status = _func_get_name(&name);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get plugin name. Status code: " + std::to_string(status)
                                   + ", Error: " + std::string(get_last_error_string()));
    }
    return name;
}

std::string_view Plugin_base::version() const
{
    assert(_initialized);
    const char* version;
    auto status = _func_get_version(&version);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get plugin version. Status code: "
                                   + std::to_string(status)
                                   + ", Error: " + std::string(get_last_error_string()));
    }
    return version;
}

hipdnnPluginType_t Plugin_base::type() const
{
    assert(_initialized);
    hipdnnPluginType_t type;
    auto status = _func_get_type(&type);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Failed to get plugin type. Status code: " + std::to_string(status)
                                   + ", Error: " + std::string(get_last_error_string()));
    }
    return type;
}

std::string_view Plugin_base::get_last_error_string() const noexcept
{
    assert(_initialized);
    const char* error_str = nullptr;
    _func_get_last_error_str(&error_str);
    return error_str;
}

} // namespace plugin
} // hipdnn_backend
