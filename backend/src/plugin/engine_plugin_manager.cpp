// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <mutex>
#include <algorithm>
#include <vector>

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/engine_heuristic_descriptor.hpp"
#include "descriptors/execution_plan_descriptor.hpp"
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

std::shared_ptr<Engine_plugin_manager> Engine_plugin_manager::create()
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

    return std::make_shared<Engine_plugin_manager>(root_pm);
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

std::vector<int64_t>
    Engine_plugin_manager::get_applicable_engine_ids(Graph_descriptor* graph_desc) const
{
    const auto& serialized_graph = graph_desc->get_serialized_graph();
    const hipdnnPluginConstData_t serialized_graph_data{serialized_graph.data(),
                                                        serialized_graph.size()};

    std::vector<int64_t> engine_ids;

    for(const auto& [handle, plugin] : _handle_to_plugin)
    {
        auto ids = plugin->get_applicable_engine_ids(handle, &serialized_graph_data);
        engine_ids.insert(engine_ids.end(), ids.begin(), ids.end());

        for(const auto& id : ids)
        {
            auto it = _engine_id_to_handle.find(id);
            if(it != _engine_id_to_handle.end() && it->second != handle)
            {
                throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                                       "Engine ID " + std::to_string(id)
                                           + " is already associated with a different plugin");
            }
            _engine_id_to_handle[id] = handle;
        }
    }

    return engine_ids;
}

void Engine_plugin_manager::get_engine_details(int64_t engine_id,
                                               Graph_descriptor* graph_desc,
                                               hipdnnPluginConstData_t* engine_details) const
{
    const auto& serialized_graph = graph_desc->get_serialized_graph();
    const hipdnnPluginConstData_t serialized_graph_data{serialized_graph.data(),
                                                        serialized_graph.size()};

    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->get_engine_details(handle, engine_id, &serialized_graph_data, engine_details);

    if(engine_details->ptr == nullptr || engine_details->size == 0)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                               "Engine details for engine ID " + std::to_string(engine_id)
                                   + " are empty or null");
    }
}

void Engine_plugin_manager::destroy_engine_details(int64_t engine_id,
                                                   hipdnnPluginConstData_t* engine_details) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->destroy_engine_details(handle, engine_details);
}

std::unique_ptr<Engine_details_wrapper> get_engine_details(const std::shared_ptr<Engine_plugin_manager>& pm, int64_t engine_id, Graph_descriptor* graph_desc)
{
    return std::make_unique<Engine_details_wrapper>(pm, engine_id, graph_desc);
}

size_t Engine_plugin_manager::get_workspace_size(int64_t engine_id,
                                                 const hipdnnPluginConstData_t* engine_config,
                                                 const hipdnnPluginConstData_t* op_graph) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    return plugin->get_workspace_size(handle, engine_config, op_graph);
}

// TODO: Pack engine_config
// TODO: Get engine_id from engine_config
size_t Engine_plugin_manager::get_workspace_size(int64_t engine_id,
                                                 const hipdnnPluginConstData_t* engine_config,
                                                 Graph_descriptor* graph_desc) const
{
    const auto& serialized_graph = graph_desc->get_serialized_graph();
    const hipdnnPluginConstData_t serialized_graph_data{serialized_graph.data(),
                                                        serialized_graph.size()};
    return get_workspace_size(engine_id, engine_config, &serialized_graph_data);
}

hipdnnEnginePluginExecutionContext_t
    Engine_plugin_manager::create_execution_context(int64_t engine_id,
                                                    const hipdnnPluginConstData_t* engine_config,
                                                    const hipdnnPluginConstData_t* op_graph) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    return plugin->create_execution_context(handle, engine_config, op_graph);
}

// TODO: Pack engine_config
// TODO: Get engine_id from engine_config
// TODO: Return a smart pointer to the execution context
hipdnnEnginePluginExecutionContext_t
    Engine_plugin_manager::create_execution_context(int64_t engine_id,
                                                    const hipdnnPluginConstData_t* engine_config,
                                                    Graph_descriptor* graph_desc) const
{
    const auto& serialized_graph = graph_desc->get_serialized_graph();
    const hipdnnPluginConstData_t serialized_graph_data{serialized_graph.data(),
                                                        serialized_graph.size()};
    return create_execution_context(engine_id, engine_config, &serialized_graph_data);
}

void Engine_plugin_manager::destroy_execution_context(
    int64_t engine_id, hipdnnEnginePluginExecutionContext_t execution_context) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->destroy_execution_context(handle, execution_context);
}

void Engine_plugin_manager::execute_op_graph(int64_t engine_id,
                                             hipdnnEnginePluginExecutionContext_t execution_context,
                                             void* workspace,
                                             const hipdnnPluginDeviceBuffer_t* device_buffers,
                                             uint32_t num_device_buffers) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->execute_op_graph(
        handle, execution_context, workspace, device_buffers, num_device_buffers);
}

void Engine_plugin_manager::finalize_engine(hipdnnBackendDescriptor_t desc) const
{
    assert(desc->type == HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    auto engine_desc = static_cast<Engine_descriptor*>(desc);

    engine_desc->finalize();

    hipdnnBackendDescriptor_t graph;
    engine_desc->get_attribute(HIPDNN_ATTR_ENGINE_OPERATION_GRAPH,
                              HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                              1,
                              nullptr,
                              &graph);
    auto graph_desc = static_cast<Graph_descriptor*>(graph);

    int64_t engine_id;
    engine_desc->get_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    auto engine_ids = get_applicable_engine_ids(graph_desc);
    if (std::ranges::find(engine_ids, engine_id) == engine_ids.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine ID " + std::to_string(engine_id)
                                   + " is not in a valid range of engine IDs");
    }

    // TODO Implement getting engine details
    // This will be implemented at the integration stage
}

#if 0
void Engine_plugin_manager::finalize_engine_config(hipdnnBackendDescriptor_t desc) const
{
    // TODO: Implement finalize_engine_config
}
#endif

#if 0
void Engine_plugin_manager::finalize_engine_heuristic(hipdnnBackendDescriptor_t desc) const
{
    // TODO: Implement finalize_engine_heuristic
}
#endif

#if 0
void Engine_plugin_manager::finalize_execution_plan(hipdnnBackendDescriptor_t desc) const
{
    // TODO: Implement finalize_execution_plan
}
#endif

#if 0
void Engine_plugin_manager::execute_op_graph(hipdnnBackendDescriptor_t execution_plan, hipdnnBackendDescriptor_t variant_pack) const
{
    // TODO: Implement execute_op_graph
}
#endif

Engine_details_wrapper::Engine_details_wrapper(const std::shared_ptr<Engine_plugin_manager>& pm, int64_t engine_id, Graph_descriptor* graph_desc)
        : _pm(pm)
{
    _pm->get_engine_details(engine_id, graph_desc, &_engine_details_data);
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(_engine_details_data.ptr), _engine_details_data.size);
}

Engine_details_wrapper::~Engine_details_wrapper()
{
    if(_engine_details_data.ptr == nullptr)
    {
        return;
    }

    try
    {
        _pm->destroy_engine_details(get()->engine_id(), &_engine_details_data);
    }
    catch(const Hipdnn_exception& e)
    {
        HIPDNN_LOG_ERROR(e.get_message());
    }
}

Engine_details_wrapper::Engine_details_wrapper(Engine_details_wrapper&& other) noexcept
    : _pm(std::move(other._pm))
    , _engine_details_data(other._engine_details_data)
{
    other._pm = nullptr;
    other._engine_details_data.ptr = nullptr;
}

Engine_details_wrapper& Engine_details_wrapper::operator=(Engine_details_wrapper&& other) noexcept
{
    if(this != &other)
    {
        _pm = std::move(other._pm);
        _engine_details_data = other._engine_details_data;

        other._pm = nullptr;
        other._engine_details_data.ptr = nullptr;
    }
    return *this;
}

const hipdnn_sdk::data_objects::EngineDetails * Engine_details_wrapper::get() const
{
    if(_engine_details_data.ptr == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR, "Wrong Engine_details_wrapper usage: "
                                                           "get() called on an empty object");
    }

    return hipdnn_sdk::data_objects::GetEngineDetails(_engine_details_data.ptr);
}

} // namespace plugin
} // hipdnn_backend
