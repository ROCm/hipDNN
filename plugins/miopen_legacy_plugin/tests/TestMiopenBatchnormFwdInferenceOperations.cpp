// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <numeric>

#include <hipdnn_sdk/plugin/EnginePluginApi.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceImplementation.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/FlatbufferGraphTestUtils.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/HalfUtils.hpp>
#include <hipdnn_sdk/utilities/HipBfloat16Utils.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

#include "HipdnnEnginePluginExecutionContext.hpp"
#include "HipdnnEnginePluginHandle.hpp"
#include "common/TestOperationsCommon.hpp"

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
                              const TensorLayout& layout);

    hipdnnEnginePluginHandle_t _handle = nullptr;
};

TEST_P(BatchnormFwdInferExecuteGraphTest, RunFloatFwdBatchnormGraphNCHW)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f, TensorLayout::NCHW);
}

TEST_P(BatchnormFwdInferExecuteGraphTest, RunBfloat16FwdBatchnormGraphNCHW)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<hip_bfloat16, float>(testCase,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              1e-2_bf,
                                              TensorLayout::NCHW);
}

TEST_P(BatchnormFwdInferExecuteGraphTest, RunHalfFwdBatchnormGraphNCHW)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<half, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h, TensorLayout::NCHW);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(BatchnormFwdInferExecuteGraphTest, DISABLED_RunDoubleFwdBatchnormGraphNCHW)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<double, double>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 1e-6, TensorLayout::NCHW);
}

TEST_P(BatchnormFwdInferExecuteGraphTest, RunFloatFwdBatchnormGraphNHWC)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<float, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_FLOAT, 1e-6f, TensorLayout::NHWC);
}

TEST_P(BatchnormFwdInferExecuteGraphTest, RunBfloat16FwdBatchnormGraphNHWC)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<hip_bfloat16, float>(testCase,
                                              hipdnn_sdk::data_objects::DataType::DataType_BFLOAT16,
                                              1e-2_bf,
                                              TensorLayout::NHWC);
}

TEST_P(BatchnormFwdInferExecuteGraphTest, RunHalfFwdBatchnormGraphNHWC)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<half, float>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_HALF, 1e-2_h, TensorLayout::NHWC);
}

// TODO: Re-enable when double support is added to MIOpen plugin
TEST_P(BatchnormFwdInferExecuteGraphTest, DISABLED_RunDoubleFwdBatchnormGraphNHWC)
{
    auto testCase = GetParam();
    runFwdBatchnormGraph<double, double>(
        testCase, hipdnn_sdk::data_objects::DataType::DataType_DOUBLE, 1e-6, TensorLayout::NHWC);
}

template <typename InputType, typename IntermediateType>
void BatchnormFwdInferExecuteGraphTest::runFwdBatchnormGraph(
    Batchnorm2dTestCase testCase,
    hipdnn_sdk::data_objects::DataType inputDataType,
    InputType epsilon,
    const TensorLayout& layout)
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

    yTensor.memory().markDeviceModified();

    hipdnnEnginePluginDestroyExecutionContext(_handle, executionContext);

    Tensor<InputType> xTensorCpu(dims, layout);
    xTensorCpu.fillWithRandomValues(
        static_cast<InputType>(0.0f), static_cast<InputType>(1.0f), seed);
    Tensor<InputType> yTensorCpu(dims, layout);
    Tensor<IntermediateType> scaleTensorCpu(derivedDims);
    scaleTensorCpu.fillWithRandomValues(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
    Tensor<IntermediateType> biasTensorCpu(derivedDims);
    biasTensorCpu.fillWithRandomValues(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
    Tensor<IntermediateType> meanTensorCpu(derivedDims);
    meanTensorCpu.fillWithRandomValues(
        static_cast<IntermediateType>(0.0f), static_cast<IntermediateType>(1.0f), seed);
    Tensor<IntermediateType> varianceTensorCpu(derivedDims);
    varianceTensorCpu.fillWithRandomValues(
        static_cast<IntermediateType>(0.1f), static_cast<IntermediateType>(1.0f), seed);

    CpuFpReferenceImplementation<InputType, IntermediateType, IntermediateType> cpuRefImpl;
    cpuRefImpl.batchnormFwdInference(xTensorCpu,
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
