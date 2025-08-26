// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/plugin/engine_plugin_api.h>
#include <hipdnn_sdk/plugin/plugin_api_data_types.h>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_implementation.hpp>
#include <hipdnn_sdk/test_utilities/cpu_fp_reference_validation.hpp>
#include <hipdnn_sdk/test_utilities/flatbuffer_graph_test_utils.hpp>
#include <hipdnn_sdk/test_utilities/test_utilities.hpp>
#include <hipdnn_sdk/utilities/half_utils.hpp>
#include <hipdnn_sdk/utilities/hip_bfloat16_utils.hpp>
#include <hipdnn_sdk/utilities/tensor.hpp>

#include "common/test_operations_common.hpp"
#include "hipdnn_engine_plugin_execution_context.hpp"
#include "hipdnn_engine_plugin_handle.hpp"

using namespace hipdnn_sdk::reference_test_utilities;
using namespace test_operations_common;

class Batchnorm_fwd_infer_execute_graph_test : public ::testing::TestWithParam<Bn_2d_test_case>
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

    template <typename Input_type, typename Intermediate_type>
    // NOLINTNEXTLINE(readability-identifier-naming)
    void RunFwdBatchnormGraph(Bn_2d_test_case test_case,
                              hipdnn_sdk::data_objects::DataType input_data_type,
                              Input_type epsilon,
                              const Tensor_layout& layout);

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

TEST_P(Batchnorm_fwd_infer_execute_graph_test, RunFloatFwdBatchnormGraphNCHW)
{
    Bn_2d_test_case test_case = GetParam();
    RunFwdBatchnormGraph<float, float>(
        test_case, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f, Tensor_layout::NCHW);
}

TEST_F(Batchnorm_fwd_infer_execute_graph_test, RunBfloat16FwdBatchnormGraphNCHW)
{
    Bn_2d_test_case test_case = {.n = 1, .c = 3, .h = 14, .w = 14};
    RunFwdBatchnormGraph<hip_bfloat16, float>(test_case,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              1e-2_bf,
                                              Tensor_layout::NCHW);
}

TEST_F(Batchnorm_fwd_infer_execute_graph_test, RunHalfFwdBatchnormGraphNCHW)
{
    Bn_2d_test_case test_case = {.n = 1, .c = 3, .h = 14, .w = 14};
    RunFwdBatchnormGraph<half, float>(
        test_case, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h, Tensor_layout::NCHW);
}

TEST_P(Batchnorm_fwd_infer_execute_graph_test, RunFloatFwdBatchnormGraphNHWC)
{
    Bn_2d_test_case test_case = GetParam();
    RunFwdBatchnormGraph<float, float>(
        test_case, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f, Tensor_layout::NHWC);
}

TEST_F(Batchnorm_fwd_infer_execute_graph_test, RunBfloat16FwdBatchnormGraphNHWC)
{
    Bn_2d_test_case test_case = {.n = 1, .c = 3, .h = 14, .w = 14};
    RunFwdBatchnormGraph<hip_bfloat16, float>(test_case,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              1e-2_bf,
                                              Tensor_layout::NHWC);
}

TEST_F(Batchnorm_fwd_infer_execute_graph_test, RunHalfFwdBatchnormGraphNHWC)
{
    Bn_2d_test_case test_case = {.n = 1, .c = 3, .h = 14, .w = 14};
    RunFwdBatchnormGraph<half, float>(
        test_case, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h, Tensor_layout::NHWC);
}

// TODO: Re-enable when double support is added to MIOpen plugin
// TEST_F(Batchnorm_fwd_infer_execute_graph_test, RunDoubleFwdBatchnormGraph)
// {
//     Bn_2d_test_case test_case = {.n = 1, .c = 3, .h = 14, .w = 14};
//     RunFwdBatchnormGraph<double, double>(
//         test_case, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 1e-6);
// }

template <typename Input_type, typename Intermediate_type>
void Batchnorm_fwd_infer_execute_graph_test::RunFwdBatchnormGraph(
    Bn_2d_test_case test_case,
    hipdnn_sdk::data_objects::DataType input_data_type,
    Input_type epsilon,
    const Tensor_layout& layout)
{
    unsigned int seed = std::random_device{}();

    std::vector<int64_t> dims = {test_case.n, test_case.c, test_case.h, test_case.w};

    std::vector<int64_t> derived_dims = {1, dims[1], 1, 1};

    std::vector<hipdnnPluginDeviceBuffer_t> device_buffers;

    PinnedTensor<Input_type> x_tensor(dims, layout);
    device_buffers.push_back(generate_random_device_buffer(
        x_tensor, 1, static_cast<Input_type>(0.0f), static_cast<Input_type>(1.0f), seed));

    PinnedTensor<Input_type> y_tensor(dims, layout);
    device_buffers.push_back(generate_empty_device_buffer(y_tensor, 2));

    PinnedTensor<Intermediate_type> scale_tensor(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(scale_tensor,
                                                           3,
                                                           static_cast<Intermediate_type>(0.0f),
                                                           static_cast<Intermediate_type>(1.0f),
                                                           seed));

    PinnedTensor<Intermediate_type> bias_tensor(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(bias_tensor,
                                                           4,
                                                           static_cast<Intermediate_type>(0.0f),
                                                           static_cast<Intermediate_type>(1.0f),
                                                           seed));

    PinnedTensor<Intermediate_type> mean_tensor(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(mean_tensor,
                                                           5,
                                                           static_cast<Intermediate_type>(0.0f),
                                                           static_cast<Intermediate_type>(1.0f),
                                                           seed));

    PinnedTensor<Intermediate_type> variance_tensor(derived_dims);
    device_buffers.push_back(generate_random_device_buffer(variance_tensor,
                                                           6,
                                                           static_cast<Intermediate_type>(0.1f),
                                                           static_cast<Intermediate_type>(1.0f),
                                                           seed));

    auto batchnorm_builder = flatbuffer_test_utils::create_valid_batchnorm_graph(
        x_tensor.strides(), x_tensor.dims(), true, input_data_type);

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

    Tensor<Input_type> x_tensor_cpu(dims, layout);
    x_tensor_cpu.fill_with_random_values(
        static_cast<Input_type>(0.0f), static_cast<Input_type>(1.0f), seed);
    Tensor<Input_type> y_tensor_cpu(dims, layout);
    Tensor<Intermediate_type> scale_tensor_cpu(derived_dims);
    scale_tensor_cpu.fill_with_random_values(
        static_cast<Intermediate_type>(0.0f), static_cast<Intermediate_type>(1.0f), seed);
    Tensor<Intermediate_type> bias_tensor_cpu(derived_dims);
    bias_tensor_cpu.fill_with_random_values(
        static_cast<Intermediate_type>(0.0f), static_cast<Intermediate_type>(1.0f), seed);
    Tensor<Intermediate_type> mean_tensor_cpu(derived_dims);
    mean_tensor_cpu.fill_with_random_values(
        static_cast<Intermediate_type>(0.0f), static_cast<Intermediate_type>(1.0f), seed);
    Tensor<Intermediate_type> variance_tensor_cpu(derived_dims);
    variance_tensor_cpu.fill_with_random_values(
        static_cast<Intermediate_type>(0.1f), static_cast<Intermediate_type>(1.0f), seed);

    Cpu_fp_reference_implementation<Input_type, Intermediate_type, Intermediate_type> cpu_ref_impl;
    cpu_ref_impl.batchnorm_fwd_inference(x_tensor_cpu,
                                         scale_tensor_cpu,
                                         bias_tensor_cpu,
                                         mean_tensor_cpu,
                                         variance_tensor_cpu,
                                         y_tensor_cpu,
                                         1e-3);

    Cpu_fp_reference_validation<Input_type> cpu_ref_validation(epsilon, epsilon);
    EXPECT_TRUE(cpu_ref_validation.all_close(y_tensor_cpu.memory(), y_tensor.memory()));
}

INSTANTIATE_TEST_SUITE_P(RunFwdBatchnormGraphWithParams,
                         Batchnorm_fwd_infer_execute_graph_test,
                         testing::ValuesIn(get_bn_2d_test_cases()));
