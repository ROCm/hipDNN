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

struct Plugin_loading_config
{
    std::vector<std::filesystem::path> paths;
    hipdnnPluginLoadingMode_ext_t mode = HIPDNN_PLUGIN_LOADING_ADDITIVE;
};

std::mutex plugin_mutex;
Plugin_loading_config plugin_config;
std::weak_ptr<Engine_plugin_manager> pm_ptr;

std::vector<std::filesystem::path> get_default_plugin_paths()
{
    return {"/opt/rocm/lib/hipdnn_plugins/engine/"};
}

} // namespace

void Engine_plugin_resource_manager::set_plugin_paths(
    const std::vector<std::filesystem::path>& plugin_paths,
    hipdnnPluginLoadingMode_ext_t loading_mode)
{
    std::lock_guard<std::mutex> lock(plugin_mutex);

    THROW_IF_FALSE(pm_ptr.expired(),
                   HIPDNN_STATUS_NOT_SUPPORTED,
                   "hipdnnSetEnginePluginPaths_ext cannot be called with an active handle.");

    plugin_config.mode = loading_mode;

    if(plugin_config.mode == HIPDNN_PLUGIN_LOADING_ABSOLUTE)
    {
        plugin_config.paths = plugin_paths;
        return;
    }
    plugin_config.paths.insert(plugin_config.paths.end(), plugin_paths.begin(), plugin_paths.end());

    for(const auto& path : get_default_plugin_paths())
    {
        if(std::ranges::find(plugin_config.paths, path) == plugin_config.paths.end()
           || plugin_config.mode == HIPDNN_PLUGIN_LOADING_ADDITIVE)
        {
            plugin_config.paths.push_back(path);
        }
    }
}

std::vector<std::filesystem::path> Engine_plugin_resource_manager::get_plugin_paths()
{
    std::lock_guard<std::mutex> lock(plugin_mutex);
    if(plugin_config.paths.empty())
    {
        return get_default_plugin_paths();
    }
    return plugin_config.paths;
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
            const auto& paths_to_load
                = plugin_config.paths.empty() ? get_default_plugin_paths() : plugin_config.paths;

            pm = std::make_shared<Engine_plugin_manager>();
            pm->load_plugins(paths_to_load, plugin_config.mode);
            pm_ptr = pm;
        }
    }

    return std::make_shared<Engine_plugin_resource_manager>(pm);
}

Engine_plugin_resource_manager::Engine_plugin_resource_manager()
    : _pm(std::make_shared<Engine_plugin_manager>())
{
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
size_t
    Engine_plugin_resource_manager::get_workspace_size(int64_t engine_id,
                                                       const hipdnnPluginConstData_t* engine_config,
                                                       const Graph_descriptor* graph_desc) const
{
    // TODO : swap the serialized type to be hipdnnPluginConstData_t.
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

void Engine_plugin_resource_manager::execute_op_graph(hipdnnBackendDescriptor_t execution_plan,
                                                      hipdnnBackendDescriptor_t variant_pack) const
{
    auto execution_plan_desc = execution_plan->as_descriptor<Execution_plan_descriptor>();
    auto variant_pack_desc = variant_pack->as_descriptor<Variant_descriptor>();

    THROW_IF_FALSE(execution_plan_desc->is_finalized(),
                   HIPDNN_STATUS_BAD_PARAM,
                   "Engine_plugin_resource_manager::execute_op_graph failed: execution_plan_desc "
                   "is not finalized");

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
} // namespace hipdnn_backend