// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <functional>
#include <string_view>
#include <type_traits>
#include <vector>

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_api_enums.h>

#include "hipdnn_exception.hpp"
#include "shared_library.hpp"

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
    Shared_library _lib;

private:
    void resolve_symbols();

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

            try
            {
                Shared_library lib(path);
                Plugin plugin(std::move(lib));

                // Get the plugin name, version and type before we move it
                const auto name = plugin.name();
                const auto version = plugin.version();
                const auto type = plugin.type();

                _plugins.emplace_back(std::move(plugin));

                HIPDNN_LOG_INFO("Plugin loaded successfully: {}", path.string());
                // Print plugin name, version and type
                HIPDNN_LOG_INFO("Plugin info: name={}, version={}, type={}",
                                name,
                                version,
                                static_cast<int>(type));
            }
            catch(const Hipdnn_exception& e)
            {
                HIPDNN_LOG_ERROR("Error loading plugin: {}. {}", path.string(), e.get_message());
                // TODO For now we just print the error message and continue
                continue;
            }
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
