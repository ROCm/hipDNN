// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file contains the implementation of a simple plugin for the engine.
// It is used to test the plugin system, ensure that plugins can be loaded and
// unloaded correctly, and run the engine from the plugin.

#include <cstdint> // for uint32_t

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/logging/logger.hpp>
#include <hipdnn_sdk/plugin/plugin_api.h>

namespace
{

const char* const PLUGIN_NAME = "EnginePlugin1";
const char* const PLUGIN_VERSION = "1.0";
const hipdnnPluginType_t PLUGIN_TYPE = hipdnnPluginTypeEngine;
const unsigned PLUGIN_NUM_ENGINES = 1;

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
        HIPDNN_LOG_ERROR("Error - kernel launch failed. Message: {}", hipGetErrorString(error));
        return hipdnnPluginInternalError;
    }
    return hipdnnPluginStatusSuccess;
}

} // namespace

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    *name = PLUGIN_NAME;
    return hipdnnPluginStatusSuccess;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    *version = PLUGIN_VERSION;
    return hipdnnPluginStatusSuccess;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    *type = PLUGIN_TYPE;
    return hipdnnPluginStatusSuccess;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetNumEngines(unsigned* num_engines)
{
    if(num_engines == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    *num_engines = PLUGIN_NUM_ENGINES;
    return hipdnnPluginStatusSuccess;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginRunEngine(unsigned engine_index,
                                                      const uint32_t* input,
                                                      uint32_t* output,
                                                      uint32_t size)
{
    if(input == nullptr || output == nullptr)
    {
        return hipdnnPluginStatusBadParam;
    }
    if(size == 0)
    {
        return hipdnnPluginStatusSuccess; // Nothing to do
    }
    if(engine_index >= PLUGIN_NUM_ENGINES)
    {
        return hipdnnPluginInvalidValue; // Invalid engine number
    }
    return run_engine(input, output, size);
}
