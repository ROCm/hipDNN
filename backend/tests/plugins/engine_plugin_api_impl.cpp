// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file is part of the test plugin implementation.
// It contains the API functions for the test engine plugin.

#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

#include "engine_plugin_api_impl.hpp"
#include "engine_plugin_handle.hpp"

using namespace hipdnn_plugin;

namespace
{

hipdnnPluginStatus_t check_handle_validity(hipdnnEnginePluginHandle_t handle)
{
    if(handle == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                         "check_handle_validity: handle is null");
    }

    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // namespace

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    if(handle == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "hipdnnEnginePluginCreate: handle is null");
    }

    try
    {
        *handle = new hipdnnEnginePluginHandle(nullptr);
    }
    catch(const std::bad_alloc&)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_ALLOC_FAILED,
            "hipdnnEnginePluginCreate: memory allocation failed");
    }

    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    delete handle;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                            hipStream_t stream)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    handle->stream = stream;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* op_graph,
                                             int64_t* engine_ids,
                                             uint32_t max_engines,
                                             uint32_t* num_engines)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    if(op_graph == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetApplicableEngineIds: op_graph is null");
    }

    if(engine_ids == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetApplicableEngineIds: engine_ids is null");
    }

    if(max_engines == 0)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetApplicableEngineIds: max_engines is zero");
    }

    if(num_engines == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetApplicableEngineIds: num_engines is null");
    }

    return get_applicable_engine_ids(handle, op_graph, engine_ids, max_engines, num_engines);
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                       int64_t engine_id,
                                       const hipdnnPluginConstData_t* op_graph,
                                       hipdnnPluginConstData_t* engine_details)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    if(!check_engine_id_validity(engine_id))
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetEngineDetails: invalid engine_id");
    }

    if(op_graph == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "hipdnnEnginePluginGetEngineDetails: op_graph is null");
    }

    if(engine_details == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetEngineDetails: engine_details is null");
    }

    return get_engine_details(handle, engine_id, op_graph, engine_details);
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                           hipdnnPluginConstData_t* engine_details)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    if(engine_details == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginDestroyEngineDetails: engine_details is null");
    }

    return destroy_engine_details(handle, engine_details);
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engine_config,
                                       const hipdnnPluginConstData_t* op_graph,
                                       size_t* workspace_size)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    if(engine_config == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetWorkspaceSize: engine_config is null");
    }

    if(op_graph == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM, "hipdnnEnginePluginGetWorkspaceSize: op_graph is null");
    }

    if(workspace_size == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginGetWorkspaceSize: workspace_size is null");
    }

    return get_workspace_size(handle, engine_config, op_graph, workspace_size);
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engine_config,
    const hipdnnPluginConstData_t* op_graph,
    hipdnnEnginePluginExecutionContext_t* execution_context)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    if(engine_config == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginCreateExecutionContext: engine_config is null");
    }

    if(op_graph == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginCreateExecutionContext: op_graph is null");
    }

    if(execution_context == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginCreateExecutionContext: execution_context is null");
    }

    return create_execution_context(handle, engine_config, op_graph, execution_context);
}

extern "C" hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    if(execution_context == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginDestroyExecutionContext: execution_context is null");
    }

    return destroy_execution_context(handle, execution_context);
}

extern "C" hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t execution_context,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* device_buffers,
                                     uint32_t num_device_buffers)
{
    auto status = check_handle_validity(handle);
    if(status != HIPDNN_PLUGIN_STATUS_SUCCESS)
    {
        return status;
    }

    if(execution_context == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginExecuteOpGraph: execution_context is null");
    }

    // Workspace can be null if the workspace size is zero.

    if(device_buffers == nullptr)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginExecuteOpGraph: device_buffers is null");
    }

    if(num_device_buffers == 0)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_BAD_PARAM,
            "hipdnnEnginePluginExecuteOpGraph: num_device_buffers is zero");
    }

    return execute_op_graph(
        handle, execution_context, workspace, device_buffers, num_device_buffers);
}
