// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <mutex>
#include <vector>

#include <hipdnn_sdk/data_objects/engine_config_generated.h>
#include <hipdnn_sdk/data_objects/engine_details_generated.h>

#include "descriptors/engine_config_descriptor.hpp"
#include "descriptors/engine_descriptor.hpp"
#include "descriptors/engine_heuristic_descriptor.hpp"
#include "descriptors/execution_plan_descriptor.hpp"
#include "descriptors/graph_descriptor.hpp"
#include "descriptors/variant_descriptor.hpp"
#include "engine_plugin.hpp"
#include "engine_plugin_resource_manager.hpp"
#include "hipdnn_exception.hpp"

namespace hipdnn_backend
{
namespace plugin
{

class Engine_plugin_manager : public Plugin_manager_base<Engine_plugin>
{
};

namespace
{

std::mutex plugin_mutex;
std::vector<std::filesystem::path> override_plugin_paths;
std::weak_ptr<Engine_plugin_manager> pm_ptr;

std::vector<std::filesystem::path> get_default_plugin_paths()
{
    // This function should return the default plugin paths.
    // For now, we return an empty vector.
    // TODO: Implement logic to retrieve default plugin paths.
    return {};
}

} // namespace

void Engine_plugin_resource_manager::set_plugin_paths(
    const std::vector<std::filesystem::path>& plugin_paths)
{
    std::lock_guard<std::mutex> lock(plugin_mutex);

    // Check if the plugin paths are already saved, if so, do nothing.
    if(!override_plugin_paths.empty())
    {
        return;
    }

    override_plugin_paths = plugin_paths;
}

std::shared_ptr<Engine_plugin_resource_manager> Engine_plugin_resource_manager::create()
{
    auto pm = pm_ptr.lock();

    if(!pm)
    {
        std::lock_guard<std::mutex> lock(plugin_mutex);

        pm = pm_ptr.lock();

        if(!pm)
        {
            auto paths = override_plugin_paths.empty() ? get_default_plugin_paths()
                                                       : override_plugin_paths;
            pm = std::make_shared<Engine_plugin_manager>();
            pm->load_plugins(paths);
            pm_ptr = pm;
        }
    }

    return std::make_shared<Engine_plugin_resource_manager>(pm);
}

Engine_plugin_resource_manager::Engine_plugin_resource_manager(
    std::shared_ptr<Engine_plugin_manager>& pm)
    : _pm(pm)
{
    // Create plugin handles
    const auto& plugins = _pm->get_plugins();
    for(const auto& plugin : plugins)
    {
        auto handle = plugin.create_handle();

        if(_handle_to_plugin.contains(handle))
        {
            throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Plugin handle already exists");
        }

        _handle_to_plugin[handle] = &plugin;
    }
}

Engine_plugin_resource_manager::~Engine_plugin_resource_manager()
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

Engine_plugin_resource_manager::Engine_plugin_resource_manager(
    Engine_plugin_resource_manager&& other) noexcept
    : _pm(std::move(other._pm))
    , _handle_to_plugin(std::move(other._handle_to_plugin))
    , _engine_id_to_handle(std::move(other._engine_id_to_handle))
{
}

Engine_plugin_resource_manager&
    Engine_plugin_resource_manager::operator=(Engine_plugin_resource_manager&& other) noexcept
{
    if(this != &other)
    {
        _pm = std::move(other._pm);
        _handle_to_plugin = std::move(other._handle_to_plugin);
        _engine_id_to_handle = std::move(other._engine_id_to_handle);
    }
    return *this;
}

void Engine_plugin_resource_manager::set_stream(hipStream_t stream) const
{
    for(const auto& [handle, plugin] : _handle_to_plugin)
    {
        plugin->set_stream(handle, stream);
    }
}

std::vector<int64_t>
    Engine_plugin_resource_manager::get_applicable_engine_ids(Graph_descriptor* graph_desc) const
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

void Engine_plugin_resource_manager::get_engine_details(
    int64_t engine_id, Graph_descriptor* graph_desc, hipdnnPluginConstData_t* engine_details) const
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

void Engine_plugin_resource_manager::destroy_engine_details(
    int64_t engine_id, hipdnnPluginConstData_t* engine_details) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->destroy_engine_details(handle, engine_details);
}

std::unique_ptr<Engine_details_wrapper>
    get_engine_details(const std::shared_ptr<Engine_plugin_resource_manager>& rm,
                       int64_t engine_id,
                       Graph_descriptor* graph_desc)
{
    return std::make_unique<Engine_details_wrapper>(rm, engine_id, graph_desc);
}

// TODO: Pack engine_config
// TODO: Get engine_id from engine_config
size_t
    Engine_plugin_resource_manager::get_workspace_size(int64_t engine_id,
                                                       const hipdnnPluginConstData_t* engine_config,
                                                       Graph_descriptor* graph_desc) const
{
    const auto& serialized_graph = graph_desc->get_serialized_graph();
    const hipdnnPluginConstData_t serialized_graph_data{serialized_graph.data(),
                                                        serialized_graph.size()};

    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    return plugin->get_workspace_size(handle, engine_config, &serialized_graph_data);
}

// TODO: Pack engine_config
// TODO: Get engine_id from engine_config
hipdnnEnginePluginExecutionContext_t Engine_plugin_resource_manager::create_execution_context(
    int64_t engine_id,
    const hipdnnPluginConstData_t* engine_config,
    Graph_descriptor* graph_desc) const
{
    const auto& serialized_graph = graph_desc->get_serialized_graph();
    const hipdnnPluginConstData_t serialized_graph_data{serialized_graph.data(),
                                                        serialized_graph.size()};

    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    return plugin->create_execution_context(handle, engine_config, &serialized_graph_data);
}

void Engine_plugin_resource_manager::destroy_execution_context(
    int64_t engine_id, hipdnnEnginePluginExecutionContext_t execution_context) const
{
    auto handle = _engine_id_to_handle.at(engine_id);
    auto plugin = _handle_to_plugin.at(handle);

    plugin->destroy_execution_context(handle, execution_context);
}

std::unique_ptr<Engine_execution_context_wrapper>
    Engine_plugin_resource_manager::create_execution_context(
        const std::shared_ptr<Engine_plugin_resource_manager>& rm,
        int64_t engine_id,
        const hipdnnPluginConstData_t* engine_config,
        Graph_descriptor* graph_desc)
{
    return std::make_unique<Engine_execution_context_wrapper>(
        rm, engine_id, engine_config, graph_desc);
}

void Engine_plugin_resource_manager::execute_op_graph(
    int64_t engine_id,
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

#if 0
void Engine_plugin_resource_manager::finalize_engine(hipdnnBackendDescriptor_t desc) const
{
    assert(desc->type == HIPDNN_BACKEND_ENGINE_DESCRIPTOR);
    auto engine_desc = static_cast<Engine_descriptor*>(desc);

    engine_desc->finalize();

    hipdnnBackendDescriptor_t graph;
    engine_desc->get_attribute(
        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &graph);
    auto graph_desc = static_cast<Graph_descriptor*>(graph);

    int64_t engine_id;
    engine_desc->get_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    auto engine_ids = get_applicable_engine_ids(graph_desc);
    if(std::ranges::find(engine_ids, engine_id) == engine_ids.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_BAD_PARAM,
                               "Engine ID " + std::to_string(engine_id)
                                   + " is not in a valid range of engine IDs");
    }

    // TODO: Get engine details
    // This will be implemented at the integration stage
}
#endif

#if 0
void Engine_plugin_resource_manager::finalize_engine_config(hipdnnBackendDescriptor_t desc) const
{
    assert(desc->type == HIPDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    auto config_desc = static_cast<Engine_config_descriptor*>(desc);

    config_desc->finalize();

    hipdnnBackendDescriptor_t engine;
    config_desc->get_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine);

    int64_t engine_id;
    engine->get_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    hipdnnBackendDescriptor_t graph;
    engine->get_attribute(
        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &graph);
    auto graph_desc = static_cast<Graph_descriptor*>(graph);

    // TODO: Move to the engine config descriptor
    // Now we have only one parameter in the engine config, but we will add more parameters later.
    flatbuffers::FlatBufferBuilder builder;
    auto engine_config = hipdnn_sdk::data_objects::CreateEngineConfig(builder, engine_id);
    builder.Finish(engine_config);
    hipdnnPluginConstData_t engine_config_data{builder.GetBufferPointer(), builder.GetSize()};

    auto workspace_size = get_workspace_size(engine_id, &engine_config_data, graph_desc);
    // TODO: Rename set_max_workspace_size() to set_workspace_size()
    // TODO: Use size_t instead of int64_t for workspace size
    // This will be implemented at the integration stage
    config_desc->set_max_workspace_size(static_cast<int64_t>(workspace_size));
}
#endif

#if 0
void Engine_plugin_resource_manager::finalize_engine_heuristic(hipdnnBackendDescriptor_t desc) const
{
    assert(desc->type == HIPDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    auto heur_desc = static_cast<Engine_heuristic_descriptor*>(desc);

    heur_desc->finalize();

    hipdnnBackendDescriptor_t graph;
    heur_desc->get_attribute(
        HIPDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &graph);
    assert(graph != nullptr);
    auto graph_desc = static_cast<Graph_descriptor*>(graph);

    auto engine_ids = get_applicable_engine_ids(graph_desc);
    heur_desc->set_engine_ids(engine_ids);

    // TODO: Get engine details
}
#endif

#if 0
// NOLINTNEXTLINE (readability-convert-member-functions-to-static)
void Engine_plugin_resource_manager::finalize_execution_plan(hipdnnBackendDescriptor_t desc) const
{
    assert(desc->type == HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    auto exec_plan_desc = static_cast<Execution_plan_descriptor*>(desc);

    exec_plan_desc->finalize();

    hipdnnBackendDescriptor_t config;
    exec_plan_desc->get_attribute(HIPDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
                                  HIPDNN_TYPE_BACKEND_DESCRIPTOR,
                                  1,
                                  nullptr,
                                  &config);

    hipdnnBackendDescriptor_t engine;
    config->get_attribute(
        HIPDNN_ATTR_ENGINECFG_ENGINE, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &engine);

    int64_t engine_id;
    engine->get_attribute(
        HIPDNN_ATTR_ENGINE_GLOBAL_INDEX, HIPDNN_TYPE_INT64, 1, nullptr, &engine_id);

    hipdnnBackendDescriptor_t graph;
    engine->get_attribute(
        HIPDNN_ATTR_ENGINE_OPERATION_GRAPH, HIPDNN_TYPE_BACKEND_DESCRIPTOR, 1, nullptr, &graph);
    auto graph_desc = static_cast<Graph_descriptor*>(graph);

    // TODO: Move to the engine config descriptor
    // Now we have only one parameter in the engine config, but we will add more parameters later.
    flatbuffers::FlatBufferBuilder builder;
    auto engine_config = hipdnn_sdk::data_objects::CreateEngineConfig(builder, engine_id);
    builder.Finish(engine_config);
    hipdnnPluginConstData_t engine_config_data{builder.GetBufferPointer(), builder.GetSize()};

    // TODO: Get execution context
    // This will be implemented at the integration stage
    std::ignore = graph_desc;
    std::ignore = engine_config_data;
}
#endif

void Engine_plugin_resource_manager::execute_op_graph(hipdnnBackendDescriptor_t execution_plan,
                                                      hipdnnBackendDescriptor_t variant_pack) const
{
    auto execution_plan_desc = execution_plan->as_descriptor<Execution_plan_descriptor>();
    auto variant_pack_desc = variant_pack->as_descriptor<Variant_descriptor>();

    THROW_IF_NE(execution_plan_desc->type,
                HIPDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_plugin_resource_manager::execute_op_graph failed: Invalid execution plan "
                "descriptor type");

    THROW_IF_FALSE(execution_plan_desc->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_plugin_resource_manager::execute_op_graph failed: execution_plan_desc "
                   "is not finalized");

    THROW_IF_NE(variant_pack_desc->type,
                HIPDNN_BACKEND_VARIANT_PACK_DESCRIPTOR,
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_plugin_resource_manager::execute_op_graph failed: Invalid variant pack "
                "descriptor type");

    THROW_IF_FALSE(variant_pack_desc->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_plugin_resource_manager::execute_op_graph failed: variant_pack_desc is "
                   "not finalized");

    auto config = execution_plan_desc->get_engine_config();
    auto engine = config->get_engine();
    auto engine_id = engine->get_engine_id();
    void* workspace = variant_pack_desc->get_workspace();

    auto& tensor_ids = variant_pack_desc->get_tensor_ids();
    auto& tensor_pointers = variant_pack_desc->get_data_pointers();

    THROW_IF_NE(tensor_ids.size(),
                tensor_pointers.size(),
                HIPDNN_STATUS_BAD_PARAM,
                "Engine_plugin_resource_manager::execute_op_graph failed: "
                "tensor_ids and tensor_pointers must have the same size");

    std::vector<hipdnnPluginDeviceBuffer_t> device_buffers;
    device_buffers.reserve(tensor_ids.size());
    for(size_t i = 0; i < tensor_ids.size(); ++i)
    {
        hipdnnPluginDeviceBuffer_t buffer;
        buffer.uid = tensor_ids[i];
        buffer.ptr = const_cast<void*>(tensor_pointers[i]);
        device_buffers.push_back(buffer);
    }

    // TODO: Get execution context from the execution plan
    // This will be implemented at the integration stage
    hipdnnEnginePluginExecutionContext_t execution_context = nullptr;

    execute_op_graph(engine_id,
                     execution_context,
                     workspace,
                     device_buffers.data(),
                     static_cast<uint32_t>(tensor_ids.size()));
}

Engine_details_wrapper::Engine_details_wrapper(
    const std::shared_ptr<Engine_plugin_resource_manager>& rm,
    int64_t engine_id,
    Graph_descriptor* graph_desc)
    : _rm(rm)
{
    _rm->get_engine_details(engine_id, graph_desc, &_engine_details_data);
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(_engine_details_data.ptr),
                                   _engine_details_data.size);
}

Engine_details_wrapper::~Engine_details_wrapper()
{
    if(_engine_details_data.ptr == nullptr)
    {
        return;
    }

    try
    {
        _rm->destroy_engine_details(get()->engine_id(), &_engine_details_data);
    }
    catch(const Hipdnn_exception& e)
    {
        HIPDNN_LOG_ERROR(e.get_message());
    }
}

Engine_details_wrapper::Engine_details_wrapper(Engine_details_wrapper&& other) noexcept
    : _rm(std::move(other._rm))
    , _engine_details_data(other._engine_details_data)
{
    other._rm = nullptr;
    other._engine_details_data.ptr = nullptr;
}

Engine_details_wrapper& Engine_details_wrapper::operator=(Engine_details_wrapper&& other) noexcept
{
    if(this != &other)
    {
        _rm = std::move(other._rm);
        _engine_details_data = other._engine_details_data;

        other._rm = nullptr;
        other._engine_details_data.ptr = nullptr;
    }
    return *this;
}

const hipdnn_sdk::data_objects::EngineDetails* Engine_details_wrapper::get() const
{
    if(_engine_details_data.ptr == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Wrong Engine_details_wrapper usage: "
                               "get() called on an empty object");
    }

    return hipdnn_sdk::data_objects::GetEngineDetails(_engine_details_data.ptr);
}

// TODO: Use engine_id from engine_config
Engine_execution_context_wrapper::Engine_execution_context_wrapper(
    const std::shared_ptr<Engine_plugin_resource_manager>& rm,
    int64_t engine_id,
    const hipdnnPluginConstData_t* engine_config,
    Graph_descriptor* graph_desc)
    : _rm(rm)
    , _engine_id(engine_id)
{
    _execution_context = _rm->create_execution_context(engine_id, engine_config, graph_desc);
}

Engine_execution_context_wrapper::~Engine_execution_context_wrapper()
{
    if(_execution_context == nullptr)
    {
        return;
    }

    try
    {
        _rm->destroy_execution_context(_engine_id, _execution_context);
    }
    catch(const Hipdnn_exception& e)
    {
        HIPDNN_LOG_ERROR(e.get_message());
    }
}

Engine_execution_context_wrapper::Engine_execution_context_wrapper(
    Engine_execution_context_wrapper&& other) noexcept
    : _rm(std::move(other._rm))
    , _engine_id(other._engine_id)
    , _execution_context(other._execution_context)
{
    other._rm = nullptr;
    other._execution_context = nullptr;
}

Engine_execution_context_wrapper&
    Engine_execution_context_wrapper::operator=(Engine_execution_context_wrapper&& other) noexcept
{
    if(this != &other)
    {
        _rm = std::move(other._rm);
        _engine_id = other._engine_id;
        _execution_context = other._execution_context;

        other._rm = nullptr;
        other._execution_context = nullptr;
    }
    return *this;
}

hipdnnEnginePluginExecutionContext_t Engine_execution_context_wrapper::get() const
{
    if(_execution_context == nullptr)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_INTERNAL_ERROR,
                               "Wrong Engine_execution_context_wrapper usage: "
                               "get() called on an empty object");
    }

    return _execution_context;
}

} // namespace plugin
} // hipdnn_backend
