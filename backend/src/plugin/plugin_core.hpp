// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <functional>
#include <iostream>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <vector>

#include "shared_library.hpp"
#include <hipdnn_sdk/plugin/plugin_api_enums.h>

namespace hipdnn_backend
{
namespace plugin
{

// The Plugin_base is the base class for all plugins.
class Plugin_base
{
protected:
    // The constructor is protected to prevent direct instantiation of the class.
    Plugin_base(Shared_library&& lib);

public:
    Plugin_base(Plugin_base&& other) = default;
    virtual ~Plugin_base() = default;

    std::string_view name() const;
    std::string_view version() const;
    hipdnnPluginType_t type() const;

protected:
    // resolve_symbols must be called before using the plugin. It is used to resolve the symbols in the plugin library.
    virtual bool resolve_symbols();

    Shared_library _lib;

private:
#ifndef NDEBUG
    bool _initialized = false;
#endif
    hipdnnPluginStatus_t (*_func_get_name)(const char**);
    hipdnnPluginStatus_t (*_func_get_version)(const char**);
    hipdnnPluginStatus_t (*_func_get_type)(hipdnnPluginType_t*);
};

// The Plugin_manager_base is responsible for loading and unloading plugins. This class is the base class for all plugin managers.
template <class Plugin>
class Plugin_manager_base
{
    static_assert(std::is_base_of_v<Plugin_base, Plugin>,
                  "Plugin must be derived from Plugin_base");

public:
    virtual ~Plugin_manager_base() = default;

    void load_plugins(const std::vector<std::filesystem::path>& plugin_paths)
    {
        // Load plugins from the specified paths
        for(const auto& path : plugin_paths)
        {
            // TODO Check if the plugin with the same path is already loaded

            Shared_library lib;
            if(!lib.load(path))
            {
                // TODO We do not have a logger yet, so we just print to stderr
                std::cerr << "Error loading plugin: " << path << '\n';
                // TODO For now we just print the error message and continue
                continue;
            }

            Plugin plugin(std::move(lib));
            // TODO resolve_symbols() should be called in the constructor of the Plugin class
            if(!plugin.resolve_symbols())
            {
                // TODO We do not have a logger yet, so we just print to stderr
                std::cerr << "Error resolving symbols for plugin: " << path << '\n';
                // The plugin is probably not valid or not compatible with the current version of the library
                // TODO For now we just print the error message and continue
                continue;
            }

            // Get the plugin name, version and type before we move it
            std::string_view name;
            std::string_view version;
            hipdnnPluginType_t type;
            // Catch any exceptions thrown by plugin functions
            try
            {
                name = plugin.name();
                version = plugin.version();
                type = plugin.type();
            }
            // TODO we do not have an exception class yet, catch specific exception instead of std::exception when we have one
            catch(...)
            {
                // TODO Print error message and error code from the exception
                // TODO We do not have a logger yet, so we just print to stderr
                std::cerr << "Error getting plugin info: " << path << '\n';
                // We should not use the plugin if we cannot get its name, version or type
                // TODO For now we just print the error message and continue
                continue;
            }

            _plugins.emplace_back(std::move(plugin));

            // TODO We do not have a logger yet, so we just print to stdout
            std::cout << "Plugin loaded successfully: " << path << '\n';
            // Print plugin name, version and type
            // TODO We do not have a logger yet, so we just print to stdout
            std::cout << "Plugin info: name=" << name << ", version=" << version
                      << ", type=" << type << '\n';
        }
    }

    const std::vector<Plugin>& get_plugins() const
    {
        return _plugins;
    }

private:
    std::vector<Plugin> _plugins;
};

} // namespace plugin
} // hipdnn_backend
