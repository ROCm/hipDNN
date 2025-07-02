// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <iostream>
#include <miopen/miopen.h>

#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_helpers.hpp>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

#include "miopen_handle_factory.hpp"

static const char* _plugin_name = "miopen_legacy_plugin";
static const char* _plugin_version = "1.0.0";

// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char hipdnn_plugin::Plugin_last_error_manager::last_error[HIPDNN_MAX_ERROR_STRING_SIZE]
    = "";

template <typename T>
void throw_if_null(T* value)
{
    if(value == nullptr)
    {
        throw hipdnn_plugin::Hipdnn_plugin_exception(HIPDNN_PLUGIN_STATUS_BAD_PARAM,
                                                     std::string(typeid(T).name()) + " is nullptr");
    }
}

extern "C" {

hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(!name)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    *name = _plugin_name;

    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(!version)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    *version = _plugin_version;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(!type)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    *type = HIPDNN_PLUGIN_TYPE_ENGINE;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(!error_str)
        return;
    *error_str = "No error";
}

// Implementation of Engine Plugin API

////////////////////////////////////////////////////////////////////////////////////////////
// TODO Temporary functions, these are going to be removed soon.
////////////////////////////////////////////////////////////////////////////////////////////
hipdnnPluginStatus_t hipdnnPluginGetNumEngines(unsigned* num_engines)
{
    if(!num_engines)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t hipdnnPluginRunEngine(unsigned engine_index,
                                           const uint32_t* input,
                                           uint32_t* output,
                                           uint32_t size)
{
    if(!input || !output || size == 0)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}
////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////

hipdnnPluginStatus_t hipdnnEnginePluginCreate(hipdnnEnginePluginHandle_t* handle)
{
    return hipdnn_plugin::try_catch([&, api_name = __func__]() {
        throw_if_null(handle);

        miopen_legacy_plugin::Miopen_handle_factory::create_miopen_handle(handle);

        //LOG_API_SUCCESS(api_name, "created_handle={:p}", static_cast<void*>(*handle));
    });
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroy(hipdnnEnginePluginHandle_t handle)
{
    if(!handle)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t hipdnnEnginePluginSetStream(hipdnnEnginePluginHandle_t handle,
                                                 hipStream_t stream)
{
    if(!handle)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetApplicableEngineIds(hipdnnEnginePluginHandle_t handle,
                                             const hipdnnPluginConstData_t* op_graph,
                                             int64_t* engine_ids,
                                             uint32_t max_engines,
                                             uint32_t* num_engines)
{
    if(!handle || !op_graph || !engine_ids || !num_engines)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t hipdnnEnginePluginGetEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                        int64_t engine_id,
                                                        const hipdnnPluginConstData_t* op_graph,
                                                        hipdnnPluginConstData_t* engine_details)
{
    if(!handle || !op_graph || !engine_details)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyEngineDetails(hipdnnEnginePluginHandle_t handle,
                                                            hipdnnPluginConstData_t* engine_details)
{
    if(!handle || !engine_details)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t
    hipdnnEnginePluginGetWorkspaceSize(hipdnnEnginePluginHandle_t handle,
                                       const hipdnnPluginConstData_t* engine_config,
                                       const hipdnnPluginConstData_t* op_graph,
                                       size_t* workspace_size)
{
    if(!handle || !engine_config || !op_graph || !workspace_size)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    // TODO: Calculate MIOpen workspace size from flatbuffer config and graph
    *workspace_size = 0;
    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t hipdnnEnginePluginCreateExecutionContext(
    hipdnnEnginePluginHandle_t handle,
    const hipdnnPluginConstData_t* engine_config,
    const hipdnnPluginConstData_t* op_graph,
    hipdnnEnginePluginExecutionContext_t* execution_context)
{
    if(!handle || !engine_config || !op_graph || !execution_context)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    // TODO: Parse flatbuffer config and graph to create MIOpen execution context
    *execution_context = nullptr;
    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t hipdnnEnginePluginDestroyExecutionContext(
    hipdnnEnginePluginHandle_t handle, hipdnnEnginePluginExecutionContext_t execution_context)
{
    if(!handle || !execution_context)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    // TODO: Destroy MIOpen execution context and free resources
    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

hipdnnPluginStatus_t
    hipdnnEnginePluginExecuteOpGraph(hipdnnEnginePluginHandle_t handle,
                                     hipdnnEnginePluginExecutionContext_t execution_context,
                                     void* workspace,
                                     const hipdnnPluginDeviceBuffer_t* device_buffers,
                                     uint32_t num_device_buffers)
{
    if(!handle || !execution_context || !device_buffers)
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;

    // TODO: Execute MIOpen operations using execution context and device buffers
    return HIPDNN_PLUGIN_INTERNAL_ERROR;
}

} // extern "C"
