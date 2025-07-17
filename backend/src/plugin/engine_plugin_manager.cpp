// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <mutex>
#include <vector>

#include "engine_plugin.hpp"
#include "engine_plugin_manager.hpp"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class Root_engine_plugin_manager : public Plugin_manager_base<Engine_plugin>
{
};

namespace
{

std::mutex plugin_mutex;
std::vector<std::filesystem::path> override_plugin_paths;
std::weak_ptr<Root_engine_plugin_manager> root_pm_ptr;

std::vector<std::filesystem::path> get_default_plugin_paths()
{
    // This function should return the default plugin paths.
    // For now, we return an empty vector.
    // TODO: Implement logic to retrieve default plugin paths.
    return {};
}

} // namespace

void Engine_plugin_manager::set_plugin_paths(const std::vector<std::filesystem::path>& plugin_paths)
{
    std::lock_guard<std::mutex> lock(plugin_mutex);

    // Check if the plugin paths are already saved, if so, do nothing.
    if (!override_plugin_paths.empty())
    {
        return;
    }

    override_plugin_paths = plugin_paths;
}

std::unique_ptr<Engine_plugin_manager> Engine_plugin_manager::create()
{
    auto root_pm = root_pm_ptr.lock();

    if(!root_pm)
    {
        std::lock_guard<std::mutex> lock(plugin_mutex);

        root_pm = root_pm_ptr.lock();

        if(!root_pm)
        {
            auto paths = override_plugin_paths.empty() ? get_default_plugin_paths() : override_plugin_paths;
            root_pm = std::make_shared<Root_engine_plugin_manager>();
            root_pm->load_plugins(paths);
            root_pm_ptr = root_pm;
        }
    }

    return std::make_unique<Engine_plugin_manager>(root_pm);
}

void Engine_plugin_manager::destroy(std::unique_ptr<Engine_plugin_manager>& manager)
{
    manager.reset();
}

Engine_plugin_manager::Engine_plugin_manager(std::shared_ptr<Root_engine_plugin_manager>& root_pm)
    : _root_pm(root_pm)
{
    // Create plugin handles
    const auto& plugins = _root_pm->get_plugins();
    for(const auto& plugin : plugins)
    {
        auto handle = plugin.create_handle();

        if(_plugin_handles.find(handle) != _plugin_handles.end())
        {
            throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                                   "Plugin handle already exists");
        }

        _plugin_handles[handle] = &plugin;
    }
}

Engine_plugin_manager::~Engine_plugin_manager()
{
    // Destroy plugin handles
    for (const auto& [handle, plugin] : _plugin_handles)
    {
        try
        {
            plugin->destroy_handle(handle);
        }
        catch(const Hipdnn_exception& e)
        {
            HIPDNN_LOG_ERROR(e.get_message());
        }
    }
}

Engine_plugin_manager::Engine_plugin_manager(Engine_plugin_manager&& other) noexcept
    : _root_pm(std::move(other._root_pm)), _plugin_handles(std::move(other._plugin_handles))
{
}

Engine_plugin_manager& Engine_plugin_manager::operator=(Engine_plugin_manager&& other) noexcept
{
    if (this != &other)
    {
        _root_pm = std::move(other._root_pm);
        _plugin_handles = std::move(other._plugin_handles);
    }
    return *this;
}

void Engine_plugin_manager::set_stream(hipStream_t stream) const
{
    for (const auto& [handle, plugin] : _plugin_handles)
    {
        plugin->set_stream(handle, stream);
    }
}

} // namespace plugin
} // hipdnn_backend
