// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <algorithm>
#include <cassert>
#include <limits>

#include "engine_plugin.hpp"

namespace hipdnn_backend
{
namespace plugin
{

Engine_plugin::Engine_plugin(Shared_library&& lib)
    : Plugin_base(std::move(lib))
{
    resolve_symbols();
}

Engine_plugin::Engine_plugin()
{
    // This constructor is used for mocking purposes in tests.
#ifndef NDEBUG
    _initialized = true;
#endif
}

void Engine_plugin::resolve_symbols()
{
    if(type() != HIPDNN_PLUGIN_TYPE_ENGINE)
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Wrong plugin type");
    }

    const auto func_name_get_all_engine_ids = "hipdnnEnginePluginGetAllEngineIds";
    _func_get_all_engine_ids
        = _lib.get_symbol<decltype(_func_get_all_engine_ids)>(func_name_get_all_engine_ids);

    const auto func_name_create_handle = "hipdnnEnginePluginCreate";
    _func_create_handle = _lib.get_symbol<decltype(_func_create_handle)>(func_name_create_handle);

    const auto func_name_destroy_handle = "hipdnnEnginePluginDestroy";
    _func_destroy_handle
        = _lib.get_symbol<decltype(_func_destroy_handle)>(func_name_destroy_handle);

    const auto func_name_set_stream = "hipdnnEnginePluginSetStream";
    _func_set_stream = _lib.get_symbol<decltype(_func_set_stream)>(func_name_set_stream);

    const auto func_name_get_applicable_engine_ids = "hipdnnEnginePluginGetApplicableEngineIds";
    _func_get_applicable_engine_ids = _lib.get_symbol<decltype(_func_get_applicable_engine_ids)>(
        func_name_get_applicable_engine_ids);

    const auto func_name_get_engine_details = "hipdnnEnginePluginGetEngineDetails";
    _func_get_engine_details
        = _lib.get_symbol<decltype(_func_get_engine_details)>(func_name_get_engine_details);

    const auto func_name_destroy_engine_details = "hipdnnEnginePluginDestroyEngineDetails";
    _func_destroy_engine_details
        = _lib.get_symbol<decltype(_func_destroy_engine_details)>(func_name_destroy_engine_details);

    const auto func_name_get_workspace_size = "hipdnnEnginePluginGetWorkspaceSize";
    _func_get_workspace_size
        = _lib.get_symbol<decltype(_func_get_workspace_size)>(func_name_get_workspace_size);

    const auto func_name_create_execution_context = "hipdnnEnginePluginCreateExecutionContext";
    _func_create_execution_context = _lib.get_symbol<decltype(_func_create_execution_context)>(
        func_name_create_execution_context);

    const auto func_name_destroy_execution_context = "hipdnnEnginePluginDestroyExecutionContext";
    _func_destroy_execution_context = _lib.get_symbol<decltype(_func_destroy_execution_context)>(
        func_name_destroy_execution_context);

    const auto func_name_execute_op_graph = "hipdnnEnginePluginExecuteOpGraph";
    _func_execute_op_graph
        = _lib.get_symbol<decltype(_func_execute_op_graph)>(func_name_execute_op_graph);

#ifndef NDEBUG
    _initialized = true;
#endif
}

std::vector<int64_t> Engine_plugin::get_all_engine_ids() const
{
    assert(_initialized);

    if(!_all_engine_ids.empty())
    {
        return _all_engine_ids;
    }

    uint32_t num_engines = 0;
    invoke_plugin_function(
        "get number of engines", _func_get_all_engine_ids, nullptr, 0u, &num_engines);

    THROW_IF_EQ(num_engines, 0, HIPDNN_STATUS_PLUGIN_ERROR, "No engines found in the plugin");

    const uint32_t max_engines = num_engines;
    std::vector<int64_t> engine_ids(max_engines);

    invoke_plugin_function("get all engine IDs",
                           _func_get_all_engine_ids,
                           engine_ids.data(),
                           max_engines,
                           &num_engines);

    THROW_IF_NE(num_engines,
                max_engines,
                HIPDNN_STATUS_PLUGIN_ERROR,
                "Number of engines returned does not match expected count");

    std::ranges::sort(engine_ids);
    if(std::ranges::adjacent_find(engine_ids) != engine_ids.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Duplicate engine IDs found");
    }
    _all_engine_ids = engine_ids;

    return engine_ids;
}

hipdnnEnginePluginHandle_t Engine_plugin::create_handle() const
{
    assert(_initialized);
    hipdnnEnginePluginHandle_t handle;
    invoke_plugin_function("create engine plugin handle", _func_create_handle, &handle);
    return handle;
}

void Engine_plugin::destroy_handle(hipdnnEnginePluginHandle_t handle) const
{
    assert(_initialized);
    invoke_plugin_function("destroy engine plugin handle", _func_destroy_handle, handle);
}

void Engine_plugin::set_stream(hipdnnEnginePluginHandle_t handle, hipStream_t stream) const
{
    assert(_initialized);
    invoke_plugin_function("set stream for engine plugin handle", _func_set_stream, handle, stream);
}

std::vector<int64_t>
    Engine_plugin::get_applicable_engine_ids(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* op_graph) const
{
    assert(_initialized);

    if(_all_engine_ids.empty())
    {
        get_all_engine_ids();
    }

    const auto max_engines = static_cast<uint32_t>(_all_engine_ids.size());
    std::vector<int64_t> engine_ids(max_engines);
    uint32_t num_engines = 0;

    invoke_plugin_function("get applicable engine IDs",
                           _func_get_applicable_engine_ids,
                           handle,
                           op_graph,
                           engine_ids.data(),
                           max_engines,
                           &num_engines);

    if(num_engines == 0)
    {
        return {}; // No applicable engines found
    }

    THROW_IF_LT(max_engines,
                num_engines,
                HIPDNN_STATUS_PLUGIN_ERROR,
                "More applicable engines than expected");

    engine_ids.resize(num_engines);

    std::ranges::sort(engine_ids);
    if(std::ranges::adjacent_find(engine_ids) != engine_ids.end())
    {
        throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR, "Duplicate engine IDs found");
    }

    for(const auto engine_id : engine_ids)
    {
        if(std::ranges::find(_all_engine_ids, engine_id) == _all_engine_ids.end())
        {
            throw Hipdnn_exception(HIPDNN_STATUS_PLUGIN_ERROR,
                                   "Engine ID not found in the plugin's known IDs");
        }
    }

    return engine_ids;
}

void Engine_plugin::get_engine_details(hipdnnEnginePluginHandle_t handle,
                                       int64_t engine_id,
                                       const hipdnnPluginConstData_t* op_graph,
                                       hipdnnPluginConstData_t* engine_details) const
{
    assert(_initialized);
    invoke_plugin_function("get engine details",
                           _func_get_engine_details,
                           handle,
                           engine_id,
                           op_graph,
                           engine_details);
}

void Engine_plugin::destroy_engine_details(hipdnnEnginePluginHandle_t handle,
                                           hipdnnPluginConstData_t* engine_details) const

{
    assert(_initialized);
    invoke_plugin_function(
        "destroy engine details", _func_destroy_engine_details, handle, engine_details);
}

size_t Engine_plugin::get_workspace_size(hipdnnEnginePluginHandle_t handle,
                                         const hipdnnPluginConstData_t* engine_config,
                                         const hipdnnPluginConstData_t* op_graph) const
{
    assert(_initialized);
    size_t workspace_size = 0;
    invoke_plugin_function("get workspace size",
                           _func_get_workspace_size,
                           handle,
                           engine_config,
                           op_graph,
                           &workspace_size);
    return workspace_size;
}

hipdnnEnginePluginExecutionContext_t
    Engine_plugin::create_execution_context(hipdnnEnginePluginHandle_t handle,
                                            const hipdnnPluginConstData_t* engine_config,
                                            const hipdnnPluginConstData_t* op_graph) const
{
    assert(_initialized);
    hipdnnEnginePluginExecutionContext_t exec_context;
    invoke_plugin_function("create execution context",
                           _func_create_execution_context,
                           handle,
                           engine_config,
                           op_graph,
                           &exec_context);
    return exec_context;
}

void Engine_plugin::destroy_execution_context(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context) const
{
    assert(_initialized);
    invoke_plugin_function(
        "destroy execution context", _func_destroy_execution_context, handle, execution_context);
}

void Engine_plugin::execute_op_graph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t execution_context,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* device_buffers,
                                     uint32_t num_device_buffers) const
{
    assert(_initialized);
    invoke_plugin_function("execute op graph",
                           _func_execute_op_graph,
                           handle,
                           execution_context,
                           workspace,
                           device_buffers,
                           num_device_buffers);
}

} // namespace plugin
} // hipdnn_backend
