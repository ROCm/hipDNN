// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cassert>

#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{

Plugin::Plugin(boost::dll::shared_library&& lib)
    : _lib(std::move(lib))
{
}

bool Plugin::resolve_symbols()
{
    const auto func_name_get_name = "hipdnnPluginGetName";
    if(!_lib.has(func_name_get_name))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_get_name << "() not found\n";
        return false;
    }
    _func_get_name = _lib.get<const char*()>(func_name_get_name);

    const auto func_name_get_version = "hipdnnPluginGetVersion";
    if(!_lib.has(func_name_get_version))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_get_version << "() not found\n";
        return false;
    }
    _func_get_version = _lib.get<const char*()>(func_name_get_version);

    const auto func_name_get_type = "hipdnnPluginGetType";
    if(!_lib.has(func_name_get_type))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_get_type << "() not found\n";
        return false;
    }
    _func_get_type = _lib.get<hipdnnPluginType_t()>(func_name_get_type);

#ifndef NDEBUG
    _initialized = true;
#endif
    return true;
}

std::string_view Plugin::name() const
{
    assert(_initialized);
    return _func_get_name();
}

std::string_view Plugin::version() const
{
    assert(_initialized);
    return _func_get_version();
}

hipdnnPluginType_t Plugin::type() const
{
    assert(_initialized);
    return _func_get_type();
}

} // namespace plugin
} // hipdnn_backend
