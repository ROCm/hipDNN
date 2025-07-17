// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/gpu_memory.hpp>

#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

using namespace hipdnn::sdk::utilities;

class Batchnorm_execute_graph_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
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

TEST_F(Batchnorm_execute_graph_test, RunFwdbatchnormGraph)
{
    std::vector<int64_t> strides = {1, 3, 224, 224}; // always in nchw
    std::vector<int64_t> dims = {1, 3, 224, 224};
    auto batchnorm_builder = flatbuffer_test_utils::create_valid_batchnorm_graph(strides, dims);

    std::vector<hipdnnPluginDeviceBuffer_t> device_buffers;

    Gpu_memory<float> gpu_mem_x(dims);
    hipdnnPluginDeviceBuffer_t x_buffer;
    x_buffer.uid = 1;
    x_buffer.ptr = gpu_mem_x.data();
    device_buffers.push_back(x_buffer);

    Gpu_memory<float> gpu_mem_y(dims);
    hipdnnPluginDeviceBuffer_t y_buffer;
    y_buffer.uid = 2;
    y_buffer.ptr = gpu_mem_y.data();
    device_buffers.push_back(y_buffer);

    Gpu_memory<float> gpu_mem_scale(dims);
    hipdnnPluginDeviceBuffer_t scale_buffer;
    scale_buffer.uid = 3;
    scale_buffer.ptr = gpu_mem_scale.data();
    device_buffers.push_back(scale_buffer);

    Gpu_memory<float> gpu_mem_bias(dims);
    hipdnnPluginDeviceBuffer_t bias_buffer;
    bias_buffer.uid = 4;
    bias_buffer.ptr = gpu_mem_bias.data();
    device_buffers.push_back(bias_buffer);

    Gpu_memory<float> gpu_mem_mean(dims);
    hipdnnPluginDeviceBuffer_t mean_buffer;
    mean_buffer.uid = 5;
    mean_buffer.ptr = gpu_mem_mean.data();
    device_buffers.push_back(mean_buffer);

    Gpu_memory<float> gpu_mem_variance(dims);
    hipdnnPluginDeviceBuffer_t variance_buffer;
    variance_buffer.uid = 6;
    variance_buffer.ptr = gpu_mem_variance.data();
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
}
