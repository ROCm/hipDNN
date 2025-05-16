// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// This file contains the implementation of a simple plugin for the engine.
// It is used to test the plugin system, ensure that plugins can be loaded and
// unloaded correctly, and run the engine from the plugin.

#include <cstdint> // for uint32_t
#include <string>

#include <hip/hip_runtime.h>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/utilities/string_util.hpp>

namespace
{

const char* const PLUGIN_NAME = "EnginePlugin1";
const char* const PLUGIN_VERSION = "1.0";
const hipdnnPluginType_t PLUGIN_TYPE = HIPDNN_PLUGIN_TYPE_ENGINE;
const unsigned PLUGIN_NUM_ENGINES = 1;

// We cannot use std::string in thread-local storage here because it requires a thread-local storage destructor.
// This prevents the shared object (plugin) from being unloaded until the program terminates.
// NOLINTNEXTLINE(modernize-avoid-c-arrays)
thread_local char last_error_string[HIPDNN_PLUGIN_ERROR_STRING_MAX_LENGTH] = "";

void set_last_error_string(const std::string& error)
{
    hipdnn::sdk::utilities::copy_max_size_with_null_terminator(
        last_error_string, error.c_str(), sizeof(last_error_string));
}

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
        set_last_error_string("hipdnnPluginRunEngine: kernel launch failed, error: "
                              + std::string(hipGetErrorString(error)));
        return HIPDNN_PLUGIN_INTERNAL_ERROR;
    }
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

} // namespace

// Exported functions:

extern "C" hipdnnPluginStatus_t hipdnnPluginGetName(const char** name)
{
    if(name == nullptr)
    {
        set_last_error_string("hipdnnPluginGetName: name is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *name = PLUGIN_NAME;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetVersion(const char** version)
{
    if(version == nullptr)
    {
        set_last_error_string("hipdnnPluginGetVersion: version is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *version = PLUGIN_VERSION;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetType(hipdnnPluginType_t* type)
{
    if(type == nullptr)
    {
        set_last_error_string("hipdnnPluginGetType: type is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *type = PLUGIN_TYPE;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" void hipdnnPluginGetLastErrorString(const char** error_str)
{
    if(error_str == nullptr)
    {
        return;
    }
    *error_str = last_error_string;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginGetNumEngines(unsigned* num_engines)
{
    if(num_engines == nullptr)
    {
        set_last_error_string("hipdnnPluginGetNumEngines: num_engines is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    *num_engines = PLUGIN_NUM_ENGINES;
    return HIPDNN_PLUGIN_STATUS_SUCCESS;
}

extern "C" hipdnnPluginStatus_t hipdnnPluginRunEngine(unsigned engine_index,
                                                      const uint32_t* input,
                                                      uint32_t* output,
                                                      uint32_t size)
{
    if(input == nullptr || output == nullptr)
    {
        set_last_error_string("hipdnnPluginRunEngine: input or output is null");
        return HIPDNN_PLUGIN_STATUS_BAD_PARAM;
    }
    if(size == 0)
    {
        return HIPDNN_PLUGIN_STATUS_SUCCESS; // Nothing to do
    }
    if(engine_index >= PLUGIN_NUM_ENGINES)
    {
        set_last_error_string("hipdnnPluginRunEngine: engine_index is out of range");
        return HIPDNN_PLUGIN_INVALID_VALUE; // Invalid engine number
    }
    return run_engine(input, output, size);
}
