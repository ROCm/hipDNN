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

class BatchnormFwdInferExecuteGraphTest : public ::testing::TestWithParam<Batchnorm2dTestCase>
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
    void runFwdBatchnormGraph(Batchnorm2dTestCase testCase,
                              hipdnn_sdk::data_objects::DataType inputDataType,
                              InputType epsilon,
                              const Tensor_layout& layout);

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

TEST_P(BatchnormFwdInferExecuteGraphTest, RunFloatFwdBatchnormGraphNCHW)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f, Tensor_layout::NCHW);
}

TEST_F(BatchnormFwdInferExecuteGraphTest, RunBfloat16FwdBatchnormGraphNCHW)
{
    auto testCase = Batchnorm2dTestCase{.n = 1, .c = 3, .h = 14, .w = 14};
    runFwdBatchnormGraph<hip_bfloat16, float>(testCase,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              1e-2_bf,
                                              Tensor_layout::NCHW);
}

TEST_F(BatchnormFwdInferExecuteGraphTest, RunHalfFwdBatchnormGraphNCHW)
{
    auto testCase = Batchnorm2dTestCase{.n = 1, .c = 3, .h = 14, .w = 14};
    runFwdBatchnormGraph<half, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h, Tensor_layout::NCHW);
}

TEST_P(BatchnormFwdInferExecuteGraphTest, RunFloatFwdBatchnormGraphNHWC)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f, Tensor_layout::NHWC);
}

TEST_F(BatchnormFwdInferExecuteGraphTest, RunBfloat16FwdBatchnormGraphNHWC)
{
    auto testCase = Batchnorm2dTestCase{.n = 1, .c = 3, .h = 14, .w = 14};
    runFwdBatchnormGraph<hip_bfloat16, float>(testCase,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              1e-2_bf,
                                              Tensor_layout::NHWC);
}

TEST_F(BatchnormFwdInferExecuteGraphTest, RunHalfFwdBatchnormGraphNHWC)
{
    auto testCase = Batchnorm2dTestCase{.n = 1, .c = 3, .h = 14, .w = 14};
    runFwdBatchnormGraph<half, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h, Tensor_layout::NHWC);
}

// TODO: Re-enable when double support is added to MIOpen plugin
// TEST_F(BatchnormFwdInferExecuteGraphTest, RunDoubleFwdBatchnormGraph)
// {
//     auto testCase = Batchnorm2dTestCase{.n = 1, .c = 3, .h = 14, .w = 14};
//     runFwdBatchnormGraph<double, double>(
//         testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 1e-6);
// }

template <typename InputType, typename IntermediateType>
void BatchnormFwdInferExecuteGraphTest::runFwdBatchnormGraph(
    Batchnorm2dTestCase testCase,
    hipdnn_sdk::data_objects::DataType inputDataType,
    InputType epsilon,
    const Tensor_layout& layout)
{
    auto seed = std::random_device{}();

    auto dims = std::vector<int64_t>{testCase.n, testCase.c, testCase.h, testCase.w};

    auto derivedDims = std::vector<int64_t>{1, dims[1], 1, 1};

    auto deviceBuffers = std::vector<hipdnnPluginDeviceBuffer_t>{};

    PinnedTensor<InputType> xTensor(dims, layout);
    deviceBuffers.push_back(generateRandomDeviceBuffer(
        xTensor, 1, static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed));

    PinnedTensor<InputType> yTensor(dims, layout);
    deviceBuffers.push_back(generateEmptyDeviceBuffer(yTensor, 2));

    PinnedTensor<IntermediateType> scaleTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(scaleTensor,
                                                       3,
                                                       static_cast<IntermediateType>(0.0f),
                                                       static_cast<IntermediateType>(1.0f),
                                                       seed));

    PinnedTensor<IntermediateType> biasTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(biasTensor,
                                                       4,
                                                       static_cast<IntermediateType>(0.0f),
                                                       static_cast<IntermediateType>(1.0f),
                                                       seed));

    PinnedTensor<IntermediateType> meanTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(meanTensor,
                                                       5,
                                                       static_cast<IntermediateType>(0.0f),
                                                       static_cast<IntermediateType>(1.0f),
                                                       seed));

    PinnedTensor<IntermediateType> varianceTensor(derivedDims);
    deviceBuffers.push_back(generateRandomDeviceBuffer(varianceTensor,
                                                       6,
                                                       static_cast<IntermediateType>(0.1f),
                                                       static_cast<IntermediateType>(1.0f),
                                                       seed));

    auto batchnormBuilder = flatbuffer_test_utils::createValidBatchnormGraph(
        xTensor.strides(), xTensor.dims(), true, inputDataType);

    hipdnnPluginConstData_t opGraph;
    opGraph.ptr = batchnormBuilder.GetBufferPointer();
    opGraph.size = batchnormBuilder.GetSize();

    auto engineConfigBuilder = flatbuffer_test_utils::createValidEngineConfig(1);
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

    yTensor.memory().mark_device_modified();

    hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);

    Tensor<InputType> xTensorCpu(dims, layout);
    xTensorCpu.fill_with_random_values(
        static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
    Tensor<InputType> yTensorCpu(dims, layout);
    Tensor<IntermediateType> scaleTensorCpu(derivedDims);
    scaleTensorCpu.fill_with_random_values(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
    Tensor<IntermediateType> biasTensorCpu(derivedDims);
    biasTensorCpu.fill_with_random_values(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
    Tensor<IntermediateType> meanTensorCpu(derivedDims);
    meanTensorCpu.fill_with_random_values(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
    Tensor<IntermediateType> varianceTensorCpu(derivedDims);
    varianceTensorCpu.fill_with_random_values(
        static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f), seed);

    CpuFpReferenceImplementation<InputType, IntermediateType, IntermediateType> cpuRefImpl;
    cpuRefImpl.batchnorm_fwd_inference(xTensorCpu,
                                       scaleTensorCpu,
                                       biasTensorCpu,
                                       meanTensorCpu,
                                       varianceTensorCpu,
                                       yTensorCpu,
                                       1e-3);

    CpuFpReferenceValidation<InputType> cpuRefValidation(epsilon, epsilon);
    EXPECT_TRUE(cpuRefValidation.allClose(yTensorCpu.memory(), yTensor.memory()));
}

INSTANTIATE_TEST_SUITE_P(RunFwdBatchnormGraphWithParams,
                         BatchnormFwdInferExecuteGraphTest,
                         testing::ValuesIn(getBatchnorm2dTestCases()));
