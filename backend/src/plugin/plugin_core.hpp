// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "hipdnn_backend_plugin_loading_mode.h"
#include "logging/logging.hpp"
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/plugin/plugin_data_type_helpers.hpp>

#include "hipdnn_exception.hpp"
#include "shared_library.hpp"

namespace hipdnn_backend
{
namespace plugin
{

#if defined(_WIN32)
constexpr const char* SHARED_LIB_EXT = ".dll";
#else
constexpr const char* SHARED_LIB_EXT = ".so";
#endif

// The Plugin_base is the base class for all plugins.
class Plugin_base
{
protected:
    // The constructor is protected to prevent direct instantiation of the class.
    Plugin_base(Shared_library&& lib);

public:
    // Prevent copying
    Plugin_base(const Plugin_base&) = delete;
    Plugin_base& operator=(const Plugin_base&) = delete;

    // Allow moving
    Plugin_base(Plugin_base&& other) = default;
    Plugin_base& operator=(Plugin_base&& other) = default;

    virtual ~Plugin_base() = default;

    std::string_view name() const;
    std::string_view version() const;
    hipdnnPluginType_t type() const;

    static hipdnnPluginType_t get_class_type();

    hipdnnPluginStatus_t set_logging_callback(hipdnnCallback_t callback);

protected:
    // This function must not throw as it is used during error handling.
    std::string_view get_last_error_string() const noexcept;

    template <typename Callable, typename... Args>
    void invoke_plugin_function(const char* description, Callable&& func, Args&&... args) const
    {
        auto status = func(std::forward<Args>(args)...);
        if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
        {
            throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                                   std::string("Failed to ") + description + ". Status: "
                                       + to_string(status) + "(" + std::to_string(status) + ")"
                                       + ", Error: " + std::string(get_last_error_string()));
        }
    }

    Shared_library _lib;

private:
    void resolve_symbols();

#ifndef NDEBUG
    bool _initialized = false;
#endif
    hipdnnPluginStatus_t (*_func_get_name)(const char**);
    hipdnnPluginStatus_t (*_func_get_version)(const char**);
    hipdnnPluginStatus_t (*_func_get_type)(hipdnnPluginType_t*);
    void (*_func_get_last_error_str)(const char**);
    hipdnnPluginStatus_t (*_func_set_logging_callback)(hipdnnCallback_t);
};

// The Plugin_manager_base is responsible for loading and unloading plugins. This class is the base class for all plugin managers.
template <class Plugin>
class Plugin_manager_base
{
    static_assert(std::is_base_of_v<Plugin_base, Plugin>,
                  "Plugin must be derived from Plugin_base");

protected:
    explicit Plugin_manager_base(std::vector<std::filesystem::path> default_paths)
        : _default_plugin_paths(std::move(default_paths))
    {
    }

    // TODO: figure out how to ignore cognitive complexity warnings induced by logging macros
    // NOLINTBEGIN(readability-function-cognitive-complexity)
    std::vector<std::filesystem::path> resolve_default_paths() const
    {
        std::filesystem::path base_dir;
        try
        {
            base_dir = plugin::Shared_library::get_current_module_directory();
        }
        catch(const Hipdnn_exception& e)
        {
            HIPDNN_LOG_WARN(
                "Failed to resolve module directory, will use unresolved default paths: {}",
                e.get_message());
            // Fallback to using original, unresolved paths
            return _default_plugin_paths;
        }

        HIPDNN_LOG_INFO("Plugin manager: Resolving default paths relative to module directory: {}",
                        base_dir.string());

        std::vector<std::filesystem::path> resolved_paths;
        resolved_paths.reserve(_default_plugin_paths.size());

        for(const auto& path : _default_plugin_paths)
        {
            if(path.is_relative())
            {
                auto resolved = base_dir / path;
                resolved_paths.push_back(resolved);
                HIPDNN_LOG_INFO("Plugin manager: Resolved default path '{}' to '{}'",
                                path.string(),
                                resolved.string());
            }
            else
            {
                HIPDNN_LOG_INFO("Plugin manager: Using absolute default path '{}'", path.string());
                resolved_paths.push_back(path);
            }
        }

        return resolved_paths;
    }
    // NOLINTEND(readability-function-cognitive-complexity)

public:
    virtual ~Plugin_manager_base() = default;

    void load_plugins(const std::vector<std::filesystem::path>& custom_paths,
                      hipdnnPluginLoadingMode_ext_t mode)
    {
        if(mode == HIPDNN_PLUGIN_LOADING_ABSOLUTE)
        {
            clear_plugins();
            load_paths(custom_paths);
        }
        else
        {
            load_paths(custom_paths);

            auto default_paths = resolve_default_paths();
            for(const auto& path : default_paths)
            {
                if(std::filesystem::exists(path))
                {
                    scan_directory_for_plugins(path);
                }
            }
        }
    }

    const std::vector<Plugin>& get_plugins() const
    {
        return _plugins;
    }

private:
    void clear_plugins()
    {
        _plugins.clear();
        _loaded_plugin_files.clear();
    }

    void load_paths(const std::vector<std::filesystem::path>& paths)
    {
        for(const auto& path : paths)
        {
            try_load_plugin_path(path);
        }
    }

    void try_load_plugin_path(const std::filesystem::path& path)
    {
        HIPDNN_LOG_INFO("Plugin manager: Trying to load plugin path: {}", path.string());
        try
        {
            if(std::filesystem::is_directory(path))
            {
                scan_directory_for_plugins(path);
            }
            else if(path.has_filename())
            {
                load_plugin_from_file(path);
            }
        }
        catch(const std::filesystem::filesystem_error& e)
        {
            HIPDNN_LOG_WARN("Error accessing plugin path: {}. {}", path.string(), e.what());
        }
    }

    void load_plugin_from_file(const std::filesystem::path& file_path)
    {
        if(_loaded_plugin_files.contains(file_path))
        {
            return;
        }

        try
        {
            Shared_library lib(file_path);
            Plugin plugin(std::move(lib));

            const auto name = plugin.name();
            const auto version = plugin.version();
            const auto type = plugin.type();

            if(type != Plugin::get_class_type() && type != HIPDNN_PLUGIN_TYPE_UNSPECIFIED)
            {
                throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                                       std::string("Plugin type mismatch: expected ")
                                           + to_string(Plugin::get_class_type()) + ", got "
                                           + to_string(type));
            }

            _plugins.emplace_back(std::move(plugin));
            _loaded_plugin_files.insert(file_path);

            HIPDNN_LOG_INFO("Plugin loaded successfully: {}", file_path.string());
            HIPDNN_LOG_INFO("Plugin info: name={}, version={}, type={}({})",
                            name,
                            version,
                            type,
                            static_cast<int>(type));
        }
        catch(const Hipdnn_exception& e)
        {
            HIPDNN_LOG_WARN(
                "Error loading plugin from [{}]: {}", file_path.string(), e.get_message());
        }
    }

    void scan_directory_for_plugins(const std::filesystem::path& dir_path)
    {
        HIPDNN_LOG_INFO("Scanning for plugins in directory: {}", dir_path.string());
        try
        {
            for(const auto& entry : std::filesystem::directory_iterator(dir_path))
            {
                const auto& path = entry.path();
                if(entry.is_regular_file() && path.extension() == SHARED_LIB_EXT)
                {
                    HIPDNN_LOG_INFO("Plugin manager: Found potential plugin file: {}",
                                    path.string());
                    load_plugin_from_file(path);
                }
            }
        }
        catch(const std::filesystem::filesystem_error& e)
        {
            HIPDNN_LOG_WARN("Error scanning plugin directory {}: {}", dir_path.string(), e.what());
        }
    }

    std::vector<Plugin> _plugins;
    std::set<std::filesystem::path> _loaded_plugin_files;
    std::vector<std::filesystem::path> _default_plugin_paths;
};

} // namespace plugin
} // namespace hipdnn_backend