// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <filesystem>
#include <functional>
#include <set>
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

public:
    virtual ~Plugin_manager_base() = default;

    void load_plugins(const std::vector<std::filesystem::path>& search_paths,
                      hipdnnPluginLoadingMode_ext_t mode)
    {
        if(mode == HIPDNN_PLUGIN_LOADING_ABSOLUTE)
        {
            clear_plugins();
        }

        for(const auto& path : search_paths)
        {
            try
            {
                if(std::filesystem::is_directory(path))
                {
                    scan_directory_for_plugins(path, mode);
                }
                else if(std::filesystem::is_regular_file(path))
                {
                    load_plugin_from_file(path, mode);
                }
                else
                {
                    HIPDNN_LOG_WARN("Plugin path is not a file or directory, skipping: {}",
                                    path.string());
                }
            }
            catch(const std::filesystem::filesystem_error& e)
            {
                HIPDNN_LOG_WARN("Error accessing plugin path: {}. {}", path.string(), e.what());
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

    void load_plugin_from_file(const std::filesystem::path& file_path,
                               hipdnnPluginLoadingMode_ext_t mode)
    {
        try
        {
            const auto canonical_path = std::filesystem::canonical(file_path);

            if(mode != HIPDNN_PLUGIN_LOADING_ADDITIVE
               && _loaded_plugin_files.contains(canonical_path))
            {
                return;
            }

            Shared_library lib(canonical_path);
            Plugin plugin(std::move(lib));

            const auto name = plugin.name();
            const auto version = plugin.version();
            const auto type = plugin.type();

            _plugins.emplace_back(std::move(plugin));
            _loaded_plugin_files.insert(canonical_path);

            HIPDNN_LOG_INFO("Plugin loaded successfully: {}", canonical_path.string());
            HIPDNN_LOG_INFO("Plugin info: name={}, version={}, type={}({})",
                            name,
                            version,
                            type,
                            static_cast<int>(type));
        }
        catch(const Hipdnn_exception& e)
        {
            HIPDNN_LOG_WARN("Error loading plugin: {}. {}", file_path.string(), e.get_message());
        }
    }

    void scan_directory_for_plugins(const std::filesystem::path& dir_path,
                                    hipdnnPluginLoadingMode_ext_t mode)
    {
        HIPDNN_LOG_INFO("Scanning for plugins in directory: {}", dir_path.string());
        for(const auto& entry : std::filesystem::directory_iterator(dir_path))
        {
            if(entry.is_regular_file())
            {
                load_plugin_from_file(entry.path(), mode);
            }
        }
    }

    std::vector<Plugin> _plugins;
    std::set<std::filesystem::path> _loaded_plugin_files;
};

} // namespace plugin
} // namespace hipdnn_backend