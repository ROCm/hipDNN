// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.
// It contains the API functions for the test engine plugin.

#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_exception.hpp>
#include <hipdnn_sdk/plugin/plugin_helpers.hpp>

#include "engine_plugin_api_impl.hpp"
#include "engine_plugin_handle.hpp"

using namespace hipdnn_plugin;

#ifdef THROW_IF_NULL
#error "THROW_IF_NULL is already defined"
#endif
#define THROW_IF_NULL(value) \
    PLUGIN_THROW_IF_NULL(value, HIPDNN_PLUGIN_STATUS_BAD_PARAM, #value " is null")

#ifdef THROW_IF_EQ
#error "THROW_IF_EQ is already defined"
#endif
#define THROW_IF_EQ(value, expected) \
    PLUGIN_THROW_IF_EQ(value, expected, HIPDNN_PLUGIN_STATUS_INVALID_VALUE, #value " is " #expected)

namespace
{

void check_handle_validity(hipdnnEnginePluginHandle_t handle)
{
    THROW_IF_NULL(handle);
}

} // namespace

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginGetAllEngineIds(int64_t* engine_ids,
                                                                  uint32_t max_engines,
                                                                  uint32_t* num_engines)
{
    return hipdnn_plugin::try_catch([&]() {
        if(max_engines != 0)
        {
            THROW_IF_NULL(engine_ids);
        }
        THROW_IF_NULL(num_engines);

        get_all_engine_ids(engine_ids, max_engines, num_engines);
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    return hipdnn_plugin::try_catch([&]() {
        THROW_IF_NULL(handle);
        *handle = new hipdnnEnginePluginHandle(nullptr);
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        delete handle;
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                            hipStream_t stream)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        handle->stream = stream;
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* op_graph,
                                             int64_t* engine_ids,
                                             uint32_t max_engines,
                                             uint32_t* num_engines)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        THROW_IF_NULL(op_graph);
        if(max_engines != 0)
        {
            THROW_IF_NULL(engine_ids);
        }
        THROW_IF_NULL(num_engines);

        get_applicable_engine_ids(handle, op_graph, engine_ids, max_engines, num_engines);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engine_id,
                                       const hipdnnPluginConstData_t* op_graph,
                                       hipdnnPluginConstData_t* engine_details)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        check_engine_id_validity(engine_id);
        THROW_IF_NULL(op_graph);
        THROW_IF_NULL(engine_details);

        get_engine_details(handle, engine_id, op_graph, engine_details);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                           hipdnnPluginConstData_t* engine_details)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        THROW_IF_NULL(engine_details);

        destroy_engine_details(handle, engine_details);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engine_config,
                                       const hipdnnPluginConstData_t* op_graph,
                                       size_t* workspace_size)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        THROW_IF_NULL(engine_config);
        THROW_IF_NULL(op_graph);
        THROW_IF_NULL(workspace_size);

        *workspace_size = get_workspace_size(handle, engine_config, op_graph);
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engine_config,
    const hipdnnPluginConstData_t* op_graph,
    hipdnnEnginePluginExecutionContext_t* execution_context)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        THROW_IF_NULL(engine_config);
        THROW_IF_NULL(op_graph);
        THROW_IF_NULL(execution_context);

        *execution_context = create_execution_context(handle, engine_config, op_graph);
    });
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        THROW_IF_NULL(execution_context);

        destroy_execution_context(handle, execution_context);
    });
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t execution_context,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* device_buffers,
                                     uint32_t num_device_buffers)
{
    return hipdnn_plugin::try_catch([&]() {
        check_handle_validity(handle);
        THROW_IF_NULL(execution_context);
        // Workspace can be null if the workspace size is zero.
        THROW_IF_NULL(device_buffers);
        THROW_IF_EQ(num_device_buffers, 0);

        execute_op_graph(handle, execution_context, workspace, device_buffers, num_device_buffers);
    });
}
