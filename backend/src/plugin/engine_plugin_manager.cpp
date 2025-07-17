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
    if(!override_plugin_paths.empty())
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
            auto paths = override_plugin_paths.empty() ? get_default_plugin_paths()
                                                       : override_plugin_paths;
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

        if(_handle_to_plugin.find(handle) != _handle_to_plugin.end())
        {
            throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Plugin handle already exists");
        }

        _handle_to_plugin[handle] = &plugin;
    }
}

Engine_plugin_manager::~Engine_plugin_manager()
{
    // Destroy plugin handles
    for(const auto& [handle, plugin] : _handle_to_plugin)
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
    : _root_pm(std::move(other._root_pm))
    , _handle_to_plugin(std::move(other._handle_to_plugin))
    , _engine_id_to_handle(std::move(other._engine_id_to_handle))
{
}

Engine_plugin_manager& Engine_plugin_manager::operator=(Engine_plugin_manager&& other) noexcept
{
    if(this != &other)
    {
        _root_pm = std::move(other._root_pm);
        _handle_to_plugin = std::move(other._handle_to_plugin);
        _engine_id_to_handle = std::move(other._engine_id_to_handle);
    }
    return *this;
}

void Engine_plugin_manager::set_stream(hipStream_t stream) const
{
    for(const auto& [handle, plugin] : _handle_to_plugin)
    {
        plugin->set_stream(handle, stream);
    }
}

// TODO: Pack op_graph
// TODO: Combine with get_engine_details()
std::vector<int64_t> Engine_plugin_manager::get_applicable_engine_ids(const hipdnnPluginConstData_t* op_graph) const
{
    std::vector<int64_t> engine_ids;

    for(const auto& [handle, plugin] : _handle_to_plugin)
    {
        auto ids = plugin->get_applicable_engine_ids(handle, op_graph);
        engine_ids.insert(engine_ids.end(), ids.begin(), ids.end());

        for(const auto& id : ids)
        {
            auto it = _engine_id_to_handle.find(id);
            if(it != _engine_id_to_handle.end() && it->second != handle)
            {
                throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                    "Engine ID " + std::to_string(id) + " is already associated with a different plugin");
            }
            _engine_id_to_handle[id] = handle;
        }
    }

    return engine_ids;
}

// TODO: Pack op_graph
// TODO: Return a smart pointer to the engine details
// TODO: Combine with get_applicable_engine_ids()
void Engine_plugin_manager::get_engine_details(int64_t engine_id, const hipdnnPluginConstData_t* op_graph, hipdnnPluginConstData_t* engine_details) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->get_engine_details(handle, engine_id, op_graph, engine_details);
}

// TODO: Get engine_id from engine_details
void Engine_plugin_manager::destroy_engine_details(int64_t engine_id, hipdnnPluginConstData_t* engine_details) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->destroy_engine_details(handle, engine_details);
}

// TODO: Pack op_graph
// TODO: Pack engine_config
// TODO: Get engine_id from engine_config
size_t Engine_plugin_manager::get_workspace_size(int64_t engine_id, const hipdnnPluginConstData_t* engine_config, const hipdnnPluginConstData_t* op_graph) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    return plugin->get_workspace_size(handle, engine_config, op_graph);
}

// TODO: Pack op_graph
// TODO: Pack engine_config
// TODO: Get engine_id from engine_config
// TODO: Return a smart pointer to the execution context
hipdnnEnginePluginExecutionContext_t Engine_plugin_manager::create_execution_context(int64_t engine_id, const hipdnnPluginConstData_t* engine_config, const hipdnnPluginConstData_t* op_graph) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    return plugin->create_execution_context(handle, engine_config, op_graph);
}

void Engine_plugin_manager::destroy_execution_context(int64_t engine_id, hipdnnEnginePluginExecutionContext_t execution_context) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->destroy_execution_context(handle, execution_context);
}

void Engine_plugin_manager::execute_op_graph(int64_t engine_id, hipdnnEnginePluginExecutionContext_t execution_context, void* workspace, const hipdnnPluginDeviceBuffer_t* device_buffers, uint32_t num_device_buffers) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->execute_op_graph(handle, execution_context, workspace, device_buffers, num_device_buffers);
}

} // namespace plugin
} // hipdnn_backend
