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

    // Logging callback is optional
    try
    {
        const auto func_name_set_logging_callback = "hipdnnPluginSetLoggingCallback";
        _func_set_logging_callback
            = _lib.get_symbol<decltype(_func_set_logging_callback)>(func_name_set_logging_callback);
    }
    catch(const Hipdnn_exception&)
    {
        _func_set_logging_callback = nullptr;

        // Add name of plugin if ever possible
        HIPDNN_LOG_INFO("Plugin does not support logging callback");
    }

#ifndef NDEBUG
    _initialized = true;
#endif
}

std::string_view Plugin_base::name() const
{
    assert(_initialized);
    const char* name;
    invoke_plugin_function("get plugin name", _func_get_name, &name);
    return name;
}

std::string_view Plugin_base::version() const
{
    assert(_initialized);
    const char* version;
    invoke_plugin_function("get plugin version", _func_get_version, &version);
    return version;
}

hipdnnPluginType_t Plugin_base::type() const
{
    assert(_initialized);
    hipdnnPluginType_t type;
    invoke_plugin_function("get plugin type", _func_get_type, &type);
    return type;
}

std::string_view Plugin_base::get_last_error_string() const noexcept
{
    assert(_initialized);
    const char* error_str = nullptr;
    _func_get_last_error_str(&error_str);
    return error_str;
}

hipdnnPluginStatus_t Plugin_base::set_logging_callback(hipdnnCallback_t callback)
{
    assert(_initialized);
    if(_func_set_logging_callback == nullptr)
    {
        // Plugin does not support logging callback, so we vacuously return success
        return HIPDNN_PLUGIN_STATUS_SUCCESS;
    }

    return _func_set_logging_callback(callback);
}

} // namespace plugin
} // hipdnn_backend
