// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file contains the implementation of a simple engine plugin.
// It is used to test the plugin subsystem.

#include <tuple>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/plugin_last_error_manager.hpp>

#include "engine_plugin_api_impl.hpp"
#include "plugin_api_impl.hpp"

using namespace hipdnn_plugin;

const char* const PLUGIN_NAME = "EnginePlugin1";
const char* const PLUGIN_VERSION = "1.0";
const hipdnnPluginType_t PLUGIN_TYPE = HIPDNN_PLUGIN_TYPE_ENGINE;

struct hipdnnEnginePluginExecutionContext
{
    uint64_t dummy; // Placeholder
};

namespace
{

const uint32_t PLUGIN_NUM_ENGINES = 3;
const int64_t PLUGIN_FIRST_ENGINE_ID = 100;

// TODO Use op_graph instead of hardcoded value
const unsigned GPU_DATA_SIZE = 512;

// TODO Use HIP RTC to compile the kernel at runtime
__global__ void engine_kernel(const uint32_t* input, uint32_t* output, uint32_t size)
{
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < size)
    {
        output[tid] = input[tid];
    }
}

// Run the kernel
hipdnnPluginStatus_t run_engine(const uint32_t* input, uint32_t* output, uint32_t size)
{
    const auto block_size = 256U;
    const auto grid_size = (size + block_size - 1) / block_size;

    // Launch the kernel on the default stream.
    engine_kernel<<<dim3(grid_size), dim3(block_size), 0, hipStreamDefault>>>(input, output, size);

    // Check if the kernel launch was successful.
    hipError_t error = hipGetLastError();
    if(error != hipSuccess)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_INTERNAL_ERROR,
            "hipdnnPluginRunEngine: kernel launch failed, error: "
                + std::string(hipGetErrorString(error)));
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // namespace

hipdnnPluginStatus_t get_applicable_engine_ids(hipdnnEnginePluginHandle_t handle,
                                               const hipdnnPluginConstData_t* op_graph,
                                               int64_t* engine_ids,
                                               uint32_t max_engines,
                                               uint32_t* num_engines)
{
    std::ignore = handle;
    std::ignore = op_graph;

    // TODO Implement actual logic to determine applicable engine IDs.
    // Now we just return a fixed set of engine IDs.
    for(uint32_t i = 0; i < max_engines && i < PLUGIN_NUM_ENGINES; ++i)
    {
        engine_ids[i] = PLUGIN_FIRST_ENGINE_ID + i;
    }
    *num_engines = PLUGIN_NUM_ENGINES;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

bool check_engine_id_validity(int64_t engine_id)
{
    // Check if the engine_id is within the valid range.
    return (engine_id >= PLUGIN_FIRST_ENGINE_ID
            && engine_id < PLUGIN_FIRST_ENGINE_ID + PLUGIN_NUM_ENGINES);
}

hipdnnPluginStatus_t get_engine_details(hipdnnEnginePluginHandle_t handle,
                                        int64_t engine_id,
                                        const hipdnnPluginConstData_t* op_graph,
                                        hipdnnPluginConstData_t* engine_details)
{
    std::ignore = handle;
    std::ignore = engine_id;
    std::ignore = op_graph;

    // TODO Implement actual logic
    // For now, we just allocate some memory for engine details.
    size_t size = 1024;
    try
    {
        engine_details->ptr = new uint8_t[size];
    }
    catch(const std::bad_alloc&)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_ALLOC_FAILED,
            "hipdnnEnginePluginGetEngineDetails: memory allocation failed");
    }
    engine_details->size = size;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t destroy_engine_details(hipdnnEnginePluginHandle_t handle,
                                            hipdnnPluginConstData_t* engine_details)
{
    std::ignore = handle;

    delete[] static_cast<const uint8_t*>(engine_details->ptr);
    engine_details->ptr = nullptr;
    engine_details->size = 0;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t get_workspace_size(hipdnnEnginePluginHandle_t handle,
                                        const hipdnnPluginConstData_t* engine_config,
                                        const hipdnnPluginConstData_t* op_graph,
                                        size_t* workspace_size)
{
    std::ignore = handle;
    std::ignore = engine_config;
    std::ignore = op_graph;

    // TODO Implement actual logic
    // For now, we just return a fixed workspace size.
    *workspace_size = 4096;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    create_execution_context(hipdnnEnginePluginHandle_t handle,
                             const hipdnnPluginConstData_t* engine_config,
                             const hipdnnPluginConstData_t* op_graph,
                             hipdnnEnginePluginExecutionContext_t* execution_context)
{
    std::ignore = handle;
    std::ignore = engine_config;
    std::ignore = op_graph;

    // Allocate memory for the execution context.
    try
    {
        *execution_context = new hipdnnEnginePluginExecutionContext(0);
    }
    catch(const std::bad_alloc&)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_ALLOC_FAILED,
            "hipdnnEnginePluginCreateExecutionContext: memory allocation failed");
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t
    destroy_execution_context(hipdnnEnginePluginHandle_t handle,
                              hipdnnEnginePluginExecutionContext_t execution_context)
{
    std::ignore = handle;

    // Free the memory allocated for the execution context.
    delete execution_context;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

hipdnnPluginStatus_t execute_op_graph(hipdnnEnginePluginHandle_t handle,
                                      hipdnnEnginePluginExecutionContext_t execution_context,
                                      void* workspace,
                                      const hipdnnPluginDeviceBuffer_t* device_buffers,
                                      uint32_t num_device_buffers)
{
    std::ignore = handle;
    std::ignore = execution_context;
    std::ignore = workspace;

    if(num_device_buffers != 2)
    {
        return Plugin_last_error_manager::set_last_error(
            HIPDNN_PLUGIN_STATUS_INVALID_VALUE,
            "hipdnnEnginePluginExecuteOpGraph: expected 2 device buffers, got "
                + std::to_string(num_device_buffers));
    }

    return run_engine(static_cast<const uint32_t*>(device_buffers[0].ptr),
                      static_cast<uint32_t*>(device_buffers[1].ptr),
                      GPU_DATA_SIZE);
}
