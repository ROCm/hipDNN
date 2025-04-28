// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cassert>

#include "plugin_core.hpp"

namespace hipdnn_backend
{
namespace plugin
{

Plugin_base::Plugin_base(boost::dll::shared_library&& lib)
    : _lib(std::move(lib))
{
}

bool Plugin_base::resolve_symbols()
{
    const auto func_name_get_name = "hipdnnPluginGetName";
    if(!_lib.has(func_name_get_name))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_get_name << "() not found\n";
        return false;
    }
    _func_get_name = _lib.get<hipdnnPluginStatus_t(const char**)>(func_name_get_name);

    const auto func_name_get_version = "hipdnnPluginGetVersion";
    if(!_lib.has(func_name_get_version))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_get_version << "() not found\n";
        return false;
    }
    _func_get_version = _lib.get<hipdnnPluginStatus_t(const char**)>(func_name_get_version);

    const auto func_name_get_type = "hipdnnPluginGetType";
    if(!_lib.has(func_name_get_type))
    {
        // TODO We do not have a logger yet, so we just print to stderr
        std::cerr << "Error: " << func_name_get_type << "() not found\n";
        return false;
    }
    _func_get_type = _lib.get<hipdnnPluginStatus_t(hipdnnPluginType_t*)>(func_name_get_type);

#ifndef NDEBUG
    _initialized = true;
#endif
    return true;
}

std::string_view Plugin_base::name() const
{
    assert(_initialized);
    // TODO Fix formatting: indentation between the type and the variable name
    const char* name;
    auto        status = _func_get_name(&name);
    if(status != hipdnnPluginStatusSuccess)
    {
        // TODO we do not have an exception class yet, so we just throw std::runtime_error
        throw std::runtime_error("Failed to get plugin name. Status code: "
                                 + std::to_string(status));
    }
    return name;
}

std::string_view Plugin_base::version() const
{
    assert(_initialized);
    // TODO Fix formatting: indentation between the type and the variable name
    const char* version;
    auto        status = _func_get_version(&version);
    if(status != hipdnnPluginStatusSuccess)
    {
        // TODO we do not have an exception class yet, so we just throw std::runtime_error
        throw std::runtime_error("Failed to get plugin version. Status code: "
                                 + std::to_string(status));
    }
    return version;
}

hipdnnPluginType_t Plugin_base::type() const
{
    assert(_initialized);
    // TODO Fix formatting: indentation between the type and the variable name
    hipdnnPluginType_t type;
    auto               status = _func_get_type(&type);
    if(status != hipdnnPluginStatusSuccess)
    {
        // TODO we do not have an exception class yet, so we just throw std::runtime_error
        throw std::runtime_error("Failed to get plugin type. Status code: "
                                 + std::to_string(status));
    }
    return type;
}

} // namespace plugin
} // hipdnn_backend
