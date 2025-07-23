// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/gpu_memory.hpp>
#include <hipdnn_sdk/test_utilities/test_tensor.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>

#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

using namespace hipdnn::sdk::utilities;
using namespace hipdnn_sdk::reference_test_utilities;

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
    std::vector<int64_t> dims = {1, 3, 224, 224};

    std::vector<hipdnnPluginDeviceBuffer_t> device_buffers;

    Test_tensor x_tensor = Test_tensor::make_test_tensor<float>(dims);
    x_tensor.fill_with_value(1.0f); // Fill with a constant value for testing
    auto batchnorm_builder = flatbuffer_test_utils::create_valid_batchnorm_graph(x_tensor.strides(), x_tensor.dims());

    hipdnnPluginDeviceBuffer_t x_buffer;
    x_buffer.uid = 1;
    x_buffer.ptr = x_tensor.memory().device_data<float>();
    device_buffers.push_back(x_buffer);

    Test_tensor y_tensor = Test_tensor::make_test_tensor<float>(dims);
    y_tensor.fill_with_value(0.0f); // Initialize output tensor with zeros
    hipdnnPluginDeviceBuffer_t y_buffer;
    y_buffer.uid = 2;
    y_buffer.ptr = y_tensor.memory().device_data<float>();
    device_buffers.push_back(y_buffer);

    // Based on miopen::DeriveBNTensorDescriptor(), the strides for the derived tensors are
    // {1, C, 1, 1} for mean, variance, scale, and bias tensors.

    Test_tensor scale_tensor = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    scale_tensor.fill_with_value(1.0f); 
    hipdnnPluginDeviceBuffer_t scale_buffer;
    scale_buffer.uid = 3;
    scale_buffer.ptr = scale_tensor.memory().device_data<float>();
    device_buffers.push_back(scale_buffer);

    Test_tensor bias_tensor = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    bias_tensor.fill_with_value(0.1f);
    hipdnnPluginDeviceBuffer_t bias_buffer;
    bias_buffer.uid = 4;
    bias_buffer.ptr = bias_tensor.memory().device_data<float>();
    device_buffers.push_back(bias_buffer);

    Test_tensor mean_tensor = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    mean_tensor.fill_with_value(0.5f); 
    hipdnnPluginDeviceBuffer_t mean_buffer;
    mean_buffer.uid = 5;
    mean_buffer.ptr = mean_tensor.memory().device_data<float>();
    device_buffers.push_back(mean_buffer);

    Test_tensor variance_tensor = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    variance_tensor.fill_with_value(0.1f);
    hipdnnPluginDeviceBuffer_t variance_buffer;
    variance_buffer.uid = 6;
    variance_buffer.ptr = variance_tensor.memory().device_data<float>();
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

    y_tensor.memory().mark_device_modified();

    hipdnnEnginePluginDestroyExecutionContext(_handle, execution_context);

    Test_tensor x_tensor_cpu = Test_tensor::make_test_tensor<float>(dims);
    x_tensor_cpu.fill_with_value(1.0f);
    Test_tensor y_tensor_cpu = Test_tensor::make_test_tensor<float>(dims);
    y_tensor_cpu.fill_with_value(0.0f);
    Test_tensor scale_tensor_cpu = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    scale_tensor_cpu.fill_with_value(1.0f);
    Test_tensor bias_tensor_cpu = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    bias_tensor_cpu.fill_with_value(0.1f);
    Test_tensor mean_tensor_cpu = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    mean_tensor_cpu.fill_with_value(0.5f);
    Test_tensor variance_tensor_cpu = Test_tensor::make_test_tensor<float>({1, dims[1], 1, 1});
    variance_tensor_cpu.fill_with_value(0.1f);

    Cpu_fp_reference_implementation<float, float, float> cpu_ref_impl;
    cpu_ref_impl.execute(x_tensor_cpu, scale_tensor_cpu, bias_tensor_cpu, mean_tensor_cpu, variance_tensor_cpu, y_tensor_cpu, 1e-5f);

    Cpu_fp_reference_validation<float> cpu_ref_validation(0.01f, 0.01f);
    EXPECT_TRUE(cpu_ref_validation.compare_buffers(y_tensor_cpu.memory(), y_tensor.memory()));  
    //EXPECT_EQ(y_tensor.memory().host_data<float>()[0], y_tensor_cpu.memory().host_data<float>()[0]);
}
