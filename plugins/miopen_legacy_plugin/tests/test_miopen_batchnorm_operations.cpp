// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/plugin/plugin_flatbuffer_utilities.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>

#define HIP_CHECK(status)                                                                      \
    do                                                                                         \
    {                                                                                          \
        if(status != hipSuccess)                                                               \
        {                                                                                      \
            std::cerr << "HIP Error: " << hipGetErrorString(status) << " in file " << __FILE__ \
                      << " at line " << __LINE__ << "\n";                                      \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while(0)

class Batchnorm_execute_graph_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        hipdnnPluginStatus_t status = hipdnnEnginePluginCreate(&_handle);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnEnginePluginDestroy(_handle);
        }
    }

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

//todo fix complexity of test...
TEST_F(Batchnorm_execute_graph_test, RunBatchnormGraph) // NOLINT
{
    std::vector<int64_t> strides = {1, 3, 224, 224}; // always in nchw
    std::vector<int64_t> dims = {1, 3, 224, 224};
    auto batchnorm_builder = flatbuffer_test_utils::create_valid_batchnorm_graph(strides, dims);

    std::vector<hipdnnPluginDeviceBuffer_t> device_buffers;

    hipdnnPluginDeviceBuffer_t x_buffer;
    x_buffer.uid = 1;
    size_t x_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&x_buffer.ptr, static_cast<size_t>(x_buffer_size * sizeof(float))));
    device_buffers.push_back(x_buffer);

    hipdnnPluginDeviceBuffer_t y_buffer;
    y_buffer.uid = 2;
    size_t y_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&y_buffer.ptr, y_buffer_size * sizeof(float)));
    device_buffers.push_back(y_buffer);

    hipdnnPluginDeviceBuffer_t scale_buffer;
    scale_buffer.uid = 3;
    size_t scale_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&scale_buffer.ptr, scale_buffer_size * sizeof(float)));
    device_buffers.push_back(scale_buffer);

    hipdnnPluginDeviceBuffer_t bias_buffer;
    bias_buffer.uid = 4;
    size_t bias_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&bias_buffer.ptr, bias_buffer_size * sizeof(float)));
    device_buffers.push_back(bias_buffer);

    hipdnnPluginDeviceBuffer_t mean_buffer;
    mean_buffer.uid = 5;
    size_t mean_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&mean_buffer.ptr, mean_buffer_size * sizeof(float)));
    device_buffers.push_back(mean_buffer);

    hipdnnPluginDeviceBuffer_t variance_buffer;
    variance_buffer.uid = 6;
    size_t variance_buffer_size
        = std::accumulate(dims.begin(), dims.end(), size_t{1}, std::multiplies<>());
    HIP_CHECK(hipMalloc(&variance_buffer.ptr, variance_buffer_size * sizeof(float)));
    device_buffers.push_back(variance_buffer);

    hipdnnPluginConstData_t op_graph;
    op_graph.ptr = batchnorm_builder.GetBufferPointer();
    op_graph.size = batchnorm_builder.GetSize();

    auto engine_config_builder = flatbuffer_test_utils::create_valid_engine_config(1);
    hipdnnPluginConstData_t engine_config;
    engine_config.ptr = engine_config_builder.GetBufferPointer();
    engine_config.size = engine_config_builder.GetSize();

    hipdnnEnginePluginExecutionContext_t execution_context;
    hipdnnEnginePluginCreateExecutionContext(
        _handle, &engine_config, &op_graph, &execution_context);

    hipdnnPluginStatus_t status
        = hipdnnEnginePluginExecuteOpGraph(_handle,
                                           execution_context,
                                           nullptr,
                                           device_buffers.data(),
                                           static_cast<uint32_t>(device_buffers.size()));
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    hipdnnEnginePluginDestroyExecutionContext(_handle, execution_context);

    for(auto& buffer : device_buffers)
    {
        HIP_CHECK(hipFree(buffer.ptr));
    }
}
