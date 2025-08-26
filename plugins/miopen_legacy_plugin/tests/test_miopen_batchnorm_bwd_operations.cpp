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

class BatchnormBwdExecuteGraphTest : public ::testing::TestWithParam<Batchnorm2dTestCase>
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

    template <typename InputType, typename IntermediateType>
    void runBwdBatchnormGraph(Batchnorm2dTestCase testCase,
                              hipdnn_sdk::data_objects::DataType inputDataType,
                              InputType epsilon,
                              const Tensor_layout& layout);

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

TEST_P(BatchnormBwdExecuteGraphTest, RunFloatBwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 4e-3f, Tensor_layout::NCHW);
}

TEST_P(BatchnormBwdExecuteGraphTest, RunBfloat16BwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = {.n = 1, .c = 3, .h = 14, .w = 14};
    runBwdBatchnormGraph<hip_bfloat16, float>(testCase,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              4e-3_bf,
                                              Tensor_layout::NCHW);
}

TEST_P(BatchnormBwdExecuteGraphTest, RunHalfBwdBatchnormGraphNCHW)
{
    Batchnorm2dTestCase testCase = {.n = 1, .c = 3, .h = 14, .w = 14};
    runBwdBatchnormGraph<half, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 4e-3_h, Tensor_layout::NCHW);
}

TEST_P(BatchnormBwdExecuteGraphTest, RunFloatBwdBatchnormGraphNHWC)
{
    Batchnorm2dTestCase testCase = GetParam();
    runBwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 4e-3f, Tensor_layout::NHWC);
}

// TEST_P(BatchnormBwdExecuteGraphTest, RunBfloat16BwdBatchnormGraphNHWC)
// {
//     Batchnorm2dTestCase testCase = {.n = 1, .c = 3, .h = 14, .w = 14};
//     runBwdBatchnormGraph<hip_bfloat16, float>(testCase,
//                                               hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
//                                               4e-3_bf,
//                                               Tensor_layout::NHWC);
// }

// TEST_P(BatchnormBwdExecuteGraphTest, RunHalfBwdBatchnormGraphNHWC)
// {
//     Batchnorm2dTestCase testCase = {.n = 1, .c = 3, .h = 14, .w = 14};
//     runBwdBatchnormGraph<half, float>(
//         testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 4e-3_h, Tensor_layout::NHWC);
// }

// TODO: Re-enable when double support is added to MIOpen plugin
// TEST_F(BatchnormBwdExecuteGraphTest, RunDoubleBwdBatchnormGraph)
// {
//     Batchnorm2dTestCase testCase = {.n = 1, .c = 3, .h = 14, .w = 14};
//     runBwdBatchnormGraph<double, double>(
//         testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 1e-6);
// }

template <typename InputType, typename IntermediateType>
void BatchnormBwdExecuteGraphTest::runBwdBatchnormGraph(
    Batchnorm2dTestCase testCase,
    hipdnn_sdk::data_objects::DataType inputDataType,
    InputType epsilon,
    const Tensor_layout& layout)
{
    unsigned int seed = std::random_device{}();

    std::vector<int64_t> dims = {testCase.n, testCase.c, testCase.h, testCase.w};

    std::vector<int64_t> derivedDims = {1, dims[1], 1, 1};

    std::vector<hipdnnPluginDeviceBuffer_t> deviceBuffers;

    PinnedTensor<InputType> xTensor(dims, layout);
    deviceBuffers.push_back(generateRandomDeviceBuffer(
        xTensor, 1, static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed));

    PinnedTensor<InputType> dyTensor(dims, layout);
    deviceBuffers.push_back(generateRandomDeviceBuffer(
        dyTensor, 2, static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed));

    PinnedTensor<InputType> dxTensor(dims, layout);
    deviceBuffers.push_back(generateEmptyDeviceBuffer(dxTensor, 3));

    PinnedTensor<IntermediateType> scaleTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(scaleTensor,
                                                          4,
                                                          static_cast<IntermediateType>(-0.1f),
                                                          static_cast<IntermediateType>(0.1f),
                                                          seed));

    PinnedTensor<IntermediateType> dscaleTensor(derivedDims);
    deviceBuffers.push_back(generateEmptyDeviceBuffer(dscaleTensor, 5));

    PinnedTensor<IntermediateType> dbiasTensor(derivedDims);
    deviceBuffers.push_back(generateEmptyDeviceBuffer(dbiasTensor, 6));

    PinnedTensor<IntermediateType> meanTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(meanTensor,
                                                          7,
                                                          static_cast<IntermediateType>(-0.1f),
                                                          static_cast<IntermediateType>(0.1f),
                                                          seed));

    PinnedTensor<IntermediateType> invVarianceTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(invVarianceTensor,
                                                          8,
                                                          static_cast<IntermediateType>(1.9f),
                                                          static_cast<IntermediateType>(2.0f),
                                                          seed));

    auto batchnormBuilder = flatbuffer_test_utils::create_valid_batchnorm_bwd_graph(
        dyTensor.strides(), dyTensor.dims(), true, inputDataType);

    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = batchnormBuilder.GetBufferPointer();
    opGraph.size = batchnormBuilder.GetSize();

    auto engineConfigBuilder = flatbuffer_test_utils::create_valid_engine_config(1);
    hipdnnPluginConstData_t engineConfig;
    engineConfig.ptr = engineConfigBuilder.GetBufferPointer();
    engineConfig.size = engineConfigBuilder.GetSize();

    hipdnnEnginePluginExecutionContext_t executionContext;
    hipdnnEnginePluginCreateExecutionContext(_handle, &engineConfig, &opGraph, &executionContext);

    hipdnnPluginStatus_t status
        = hipdnnEnginePluginExecuteOpGraph(_handle,
                                           executionContext,
                                           nullptr,
                                           deviceBuffers.data(),
                                           static_cast<uint32_t>(deviceBuffers.size()));
    EXPECT_EQ(status, HIPDNN_PLUGIN_STATUS_SUCCESS);

    dxTensor.memory().mark_device_modified();
    dscaleTensor.memory().mark_device_modified();
    dbiasTensor.memory().mark_device_modified();

    hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);

    Tensor<InputType> xTensorCpu(dims, layout);
    xTensorCpu.fill_with_random_values(
        static_cast<InputType>(-1.0f), static_cast<InputType>(1.0f), seed);
    Tensor<InputType> dyTensorCpu(dims, layout);
    dyTensorCpu.fill_with_random_values(
        static_cast<InputType>(-0.1f), static_cast<InputType>(0.1f), seed);
    Tensor<InputType> dxTensorCpu(dims, layout);

    Tensor<IntermediateType> scaleTensorCpu(derivedDims);
    scaleTensorCpu.fill_with_random_values(
        static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);
    Tensor<IntermediateType> dscaleTensorCpu(derivedDims);
    Tensor<IntermediateType> dbiasTensorCpu(derivedDims);
    Tensor<IntermediateType> meanTensorCpu(derivedDims);
    meanTensorCpu.fill_with_random_values(
        static_cast<IntermediateType>(-0.1f), static_cast<IntermediateType>(0.1f), seed);

    Tensor<IntermediateType> invVarianceTensorCpu(derivedDims);
    invVarianceTensorCpu.fill_with_random_values(
        static_cast<IntermediateType>(1.9f), static_cast<IntermediateType>(2.0f), seed);

    Cpu_fp_reference_implementation<InputType, IntermediateType, IntermediateType> cpuRefImpl;
    cpuRefImpl.batchnorm_bwd(dyTensorCpu,
                             xTensorCpu,
                             meanTensorCpu,
                             invVarianceTensorCpu,
                             scaleTensorCpu,
                             dxTensorCpu,
                             dscaleTensorCpu,
                             dbiasTensorCpu);

    Cpu_fp_reference_validation<InputType> cpuRefValidationInput(epsilon, epsilon);
    Cpu_fp_reference_validation<IntermediateType> cpuRefValidationIntermediate(epsilon, epsilon);

    EXPECT_TRUE(cpuRefValidationInput.all_close(dxTensorCpu.memory(), dxTensor.memory()));
    EXPECT_TRUE(
        cpuRefValidationIntermediate.all_close(dscaleTensorCpu.memory(), dscaleTensor.memory()));
    EXPECT_TRUE(
        cpuRefValidationIntermediate.all_close(dbiasTensorCpu.memory(), dbiasTensor.memory()));
}

INSTANTIATE_TEST_SUITE_P(RunBwdBatchnormGraphWithParams,
                         BatchnormBwdExecuteGraphTest,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
