// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

// Use std::filesystem::path instead of boost::filesystem::path for boost::dll::fs::path
// Use std::error_code instead of boost::system::error_code for boost::dll::fs::error_code
// Use std::system_error instead of boost::system::system_error for boost::dll::fs::system_error
#define BOOST_DLL_USE_STD_FS

#include <boost/dll.hpp>
#include <filesystem>
#include <functional>
#include <iostream>
#include <string_view>
#include <system_error>
#include <vector>

#include "plugin_api.h"

namespace hipdnn_backend
{
namespace plugin
{

// The Plugin is the base class for all plugins.
class Plugin
{
protected:
    // The constructor is protected to prevent direct instantiation of the Plugin class.
    Plugin(boost::dll::shared_library&& lib);

public:
    virtual ~Plugin() = default;

    // These functions are mandatory for all plugins. They are used to get the plugin name and version.
    std::string_view   name() const;
    std::string_view   version() const;
    hipdnnPluginType_t type() const;

protected:
    // resolve_symbols must be called before using the plugin. It is used to resolve the symbols in the plugin library.
    virtual bool resolve_symbols();

    boost::dll::shared_library _lib;

private:
#ifndef NDEBUG
    bool _initialized = false; // Flag to check if the plugin is initialized
#endif
    std::function<const char*()>        _func_get_name;
    std::function<const char*()>        _func_get_version;
    std::function<hipdnnPluginType_t()> _func_get_type;
};

// The plugin manager is responsible for loading and unloading plugins. This class is the base class for all plugin managers.
template <class Plugin>
class Plugin_manager
{
public:
    virtual ~Plugin_manager() = default;

    void load_plugins(const std::vector<std::filesystem::path>& plugin_paths)
    {
        // Load plugins from the specified paths
        for(const auto& path : plugin_paths)
        {
            // TODO Check if the plugin with the same path is already loaded

            // append_decorations:
            // This mode is used to add the appropriate decorations to the library name based on the platform.
            // NOTE: if the library is not found, it will try to load the library with the exact name.
            // search_system_folders:
            // This mode allows loading from system folders if path to library contains no parent path.
            const auto mode = boost::dll::load_mode::append_decorations
                              | boost::dll::load_mode::search_system_folders;
            boost::dll::shared_library lib;
            try
            {
                lib.load(path, mode);
            }
            catch(const std::system_error& e)
            {
                // TODO We do not have a logger yet, so we just print to stderr
                std::cerr << "Error loading plugin: " << path << ", error=" << e.what()
                          << '\n';
                // TODO For now we just print the error message and continue
                continue;
            }

            Plugin plugin(std::move(lib));
            // TODO resolve_symbols() should be called in the constructor of the Plugin class
            if(!plugin.resolve_symbols())
            {
                // TODO We do not have a logger yet, so we just print to stderr
                std::cerr << "Error resolving symbols for plugin: " << path << '\n';
                // TODO For now we just print the error message and continue
                continue;
            }

            _plugins.emplace_back(std::move(plugin));

            // TODO We do not have a logger yet, so we just print to stdout
            std::cout << "Plugin loaded successfully: " << path << '\n';
        }
    }

    const std::vector<Plugin>& get_plugins() const
    {
        return _plugins;
    }

private:
    std::vector<Plugin> _plugins; // List of loaded plugins
};

} // namespace plugin
} // hipdnn_backend
