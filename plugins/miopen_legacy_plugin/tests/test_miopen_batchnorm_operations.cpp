// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>
#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_tensor.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>

#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

using namespace hipdnn_sdk::reference_test_utilities;

struct Bn_2d_test_case
{
    int64_t n;
    int64_t c;
    int64_t h;
    int64_t w;

    friend std::ostream& operator<<(std::ostream& ss, const Bn_2d_test_case& tc)
    {
        return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h << " w:" << tc.w << ")";
    }

    std::vector<int64_t> get_dims() const
    {
        return {n, c, h, w};
    }
};

class Batchnorm_execute_graph_test : public ::testing::TestWithParam<Bn_2d_test_case>
{
protected:
    // NOLINTNEXTLINE(readability-identifier-naming)
    void SetUp() override
    {
        SKIP_IF_NO_DEVICES();
        hipdnnPluginStatus_t status = hipdnnEnginePluginCreate(&_handle);
        ASSERT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);
    }

    // NOLINTNEXTLINE(readability-identifier-naming)
    void TearDown() override
    {
        if(_handle != nullptr)
        {
            hipdnnEnginePluginDestroy(_handle);
        }
    }

    template <typename Input_type>
    // NOLINTNEXTLINE(readability-identifier-naming)
    void RunFwdbatchnormGraph(Bn_2d_test_case test_case, hipdnn_sdk::data_objects::DataType input_data_type, Input_type epsilon);

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

std::vector<Bn_2d_test_case> get_bn_fwd_inference_test_cases()
{
    return {
        {.n = 1, .c = 3, .h = 14, .w = 14},
        {.n = 2, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 3, .h = 14, .w = 14},
        {.n = 64, .c = 256, .h = 14, .w = 14},
        {.n = 64, .c = 256, .h = 28, .w = 28},
        {.n = 64, .c = 256, .h = 56, .w = 56},
        {.n = 64, .c = 512, .h = 14, .w = 14},
        {.n = 64, .c = 512, .h = 28, .w = 28},
        {.n = 64, .c = 512, .h = 7, .w = 7},
        {.n = 64, .c = 64, .h = 112, .w = 112},
        {.n = 64, .c = 64, .h = 56, .w = 56},
    };
}

template <typename T>
hipdnnPluginDeviceBuffer_t
    generate_random_device_buffer(Test_tensor& tensor, int uid, T min, T max, unsigned int seed = 0)
{
    tensor.fill_with_random_values<T>(min, max, seed);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().device_data<T>();
    return buffer;
}

template <typename T>
hipdnnPluginDeviceBuffer_t generate_static_device_buffer(Test_tensor& tensor, int uid, T value)
{
    tensor.fill_with_value<T>(value);
    hipdnnPluginDeviceBuffer_t buffer;
    buffer.uid = uid;
    buffer.ptr = tensor.memory().device_data<T>();
    return buffer;
}

TEST_P(Batchnorm_execute_graph_test, RunFloatFwdbatchnormGraph)
{
    Bn_2d_test_case test_case = GetParam();
    RunFwdbatchnormGraph<float>(test_case, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f);
}

TEST_F(Batchnorm_execute_graph_test, RunBfloat16FwdbatchnormGraph)
{
    Bn_2d_test_case test_case = {.n = 1, .c = 3, .h = 14, .w = 14};
    RunFwdbatchnormGraph<hip_bfloat16>(test_case, hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16, 1e-2_bf);
}

TEST_F(Batchnorm_execute_graph_test, RunHalfFwdbatchnormGraph)
{
    Bn_2d_test_case test_case = {.n = 1, .c = 3, .h = 14, .w = 14};
    RunFwdbatchnormGraph<half>(test_case, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h);
}

template <typename Input_type>
void Batchnorm_execute_graph_test::RunFwdbatchnormGraph(Bn_2d_test_case test_case, hipdnn_sdk::data_objects::DataType input_data_type, Input_type epsilon)
{
    unsigned int seed = std::random_device{}();

    std::vector<int64_t> dims = {test_case.n, test_case.c, test_case.h, test_case.w};

    // Based on miopen::DeriveBNTensorDescriptor(), the strides for the derived tensors are
    // {1, C, 1, 1} for mean, variance, scale, and bias tensors.
    std::vector<int64_t> derived_dims = {1, dims[1], 1, 1};

    std::vector<hipdnnPluginDeviceBuffer_t> device_buffers;

    Test_tensor x_tensor = Test_tensor::make_test_tensor<Input_type>(dims);
    device_buffers.push_back(generate_random_device_buffer(x_tensor, 1, static_cast<Input_type>(0.0f), static_cast<Input_type>(1.0f), seed));

    Test_tensor y_tensor = Test_tensor::make_test_tensor<Input_type>(dims);
    device_buffers.push_back(generate_static_device_buffer(y_tensor, 2, static_cast<Input_type>(0.0f)));

    Test_tensor scale_tensor = Test_tensor::make_test_tensor<float>(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(scale_tensor, 3, 0.0f, 1.0f, seed));

    Test_tensor bias_tensor = Test_tensor::make_test_tensor<float>(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(bias_tensor, 4, 0.0f, 0.1f, seed));

    Test_tensor mean_tensor = Test_tensor::make_test_tensor<float>(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(mean_tensor, 5, 0.0f, 0.5f, seed));

    Test_tensor variance_tensor = Test_tensor::make_test_tensor<float>(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(variance_tensor, 6, 0.1f, 0.3f, seed));

    auto batchnorm_builder
        = flatbuffer_test_utils::create_valid_batchnorm_graph(x_tensor.strides(), x_tensor.dims(), true, input_data_type);

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

    Test_tensor x_tensor_cpu = Test_tensor::make_test_tensor<Input_type>(dims);
    x_tensor_cpu.fill_with_random_values(static_cast<Input_type>(0.0f), static_cast<Input_type>(1.0f), seed);
    Test_tensor y_tensor_cpu = Test_tensor::make_test_tensor<Input_type>(dims);
    y_tensor_cpu.fill_with_value(static_cast<Input_type>(0.0f));
    Test_tensor scale_tensor_cpu = Test_tensor::make_test_tensor<float>(derived_dims);
    scale_tensor_cpu.fill_with_random_values(0.0f, 1.0f, seed);
    Test_tensor bias_tensor_cpu = Test_tensor::make_test_tensor<float>(derived_dims);
    bias_tensor_cpu.fill_with_random_values(0.0f, 0.1f, seed);
    Test_tensor mean_tensor_cpu = Test_tensor::make_test_tensor<float>(derived_dims);
    mean_tensor_cpu.fill_with_random_values(0.0f, 0.5f, seed);
    Test_tensor variance_tensor_cpu = Test_tensor::make_test_tensor<float>(derived_dims);
    variance_tensor_cpu.fill_with_random_values(0.1f, 0.3f, seed);

    Cpu_fp_reference_implementation<Input_type, float, float> cpu_ref_impl;
    cpu_ref_impl.batchnorm_fwd_inference(x_tensor_cpu,
                         scale_tensor_cpu,
                         bias_tensor_cpu,
                         mean_tensor_cpu,
                         variance_tensor_cpu,
                         y_tensor_cpu,
                         1e-3f);

    Cpu_fp_reference_validation<Input_type> cpu_ref_validation(epsilon, epsilon);
    EXPECT_TRUE(cpu_ref_validation.compare_buffers(y_tensor_cpu.memory(), y_tensor.memory()));
    //EXPECT_EQ(y_tensor_cpu.memory().host_data<Input_type>()[0], y_tensor.memory().host_data<Input_type>()[0]);
}

INSTANTIATE_TEST_SUITE_P(RunFwdbatchnormGraphWithParams,
                         Batchnorm_execute_graph_test,
                         testing::ValuesIn(get_bn_fwd_inference_test_cases()));